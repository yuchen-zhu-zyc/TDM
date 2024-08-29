
import os, time, gc

import torch
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch_ema import ExponentialMovingAverage
import pytorch_warmup as warmup
import get_net
import sde
from losses import *
import data
import util
from writer import WandBWriter
import matplotlib.pyplot as plt
from ipdb import set_trace as debug

def build_optimizer_ema_sched(opt, net):
    direction = net.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer           = optim_name(net.parameters(), **optim_dict)
    ema                 = ExponentialMovingAverage(net.parameters(), decay=0.9999)
    sched               = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_itr)
    warmup_scheduler    = warmup.UntunedLinearWarmup(optimizer)

    return optimizer, ema, sched,warmup_scheduler

def freeze(net):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net

def activate(net):
    for p in net.parameters():
        p.requires_grad = True
    net.train()
    return net

class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()

        self.start_time =time.time()
        self.ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        opt.dt          = (opt.T-opt.t0)/opt.interval
        # build boundary distribution (p: target, q: prior)
        self.dists      = data.build_boundary_distribution(opt)

        # build dynamics
        self.dyn        = sde.build(opt, self.dists)
        self.net        = get_net.build(opt, self.dyn, 'backward') # q -> p
        self.optimizer, self.ema, self.sched,self.warmup_scheduler \
                        = build_optimizer_ema_sched(opt, self.net)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

        self.writer     = None
        self.logging    = False
        if opt.wandb_api_key is not None:
            self.writer = WandBWriter(opt)
            self.it     = 0
            self.logging=True
        
    def get_optimizer_ema_sched(self):
        return self.optimizer, self.ema, self.sched,self.warmup_scheduler

    @torch.no_grad()
    def sample_train_data(self, opt):
        train_ts = self.ts

        # reuse or sample training xs and zs
        gs, xis, labels = self.dyn.sample_traj(train_ts, None)
        train_gs        = gs.detach().cpu(); del gs
        train_xis       = xis.detach().cpu(); del xis
        train_labels    = labels.detach().cpu(); del labels

        print('generate train data from [{}]!'.format(util.red('sampling')))

        assert train_gs.shape[0] == opt.samp_bs
        assert train_gs.shape[1] == len(train_ts)
        gc.collect()

        return train_gs, train_xis, train_ts,train_labels

    def dsm_train(self, opt):
        net = activate(self.net)
        self.dsm_train_ep(opt,net)

    def ssm_train(self, opt):
        net = activate(self.net)
        self.ssm_train_ep(opt,net)
        
    def dsm_train_ep(self,opt,net):
        ema = self.ema
        optimizer, ema, sched,warmup= self.get_optimizer_ema_sched()
        t0,T,device                 = opt.t0, opt.T, opt.device
        bs                          = opt.train_bs_x
        net.train()
        data_sampler  = self.dists.sample_data
        self.evaluate(opt, 0)
        scaler = torch.cuda.amp.GradScaler()
        for it in range(opt.num_itr):
            optimizer.zero_grad(set_to_none=True)
            # ===== sample boundary pair =====
            x0      = data_sampler(batch_size=bs)
            # ===== compute loss =====
            _ts     = util.uniform(bs,t0,T,device)[:,None]
            gt,xit,score\
                    = self.dyn.dsm_sample(x0,_ts)
            pred    = net(gt,xit, _ts)
            label   = score.reshape_as(pred)
            reweight= self.dyn.get_sigxi(_ts)
            
            if opt.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss    = compute_DSM_train(reweight,pred,label)
            else:
                loss    = compute_DSM_train(reweight,pred,label)
            assert not torch.isnan(loss)
            
            scaler.scale(loss).backward()

            if opt.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            if sched is not None: 
                with warmup.dampening():
                    sched.step()
  
            self.log_train(opt,it+1,loss.detach().cpu(),optimizer,'backward')
            if it == 0: continue
            self.evaluate(opt, it+1)
            

    def ssm_train_ep(self, opt, net):
        net                 = activate(self.net)
        optimizer, ema, sched,warmup       \
                                    = self.get_optimizer_ema_sched()
        train_gs, train_xis, train_ts, train_labels = self.sample_train_data(opt)
        self.evaluate(opt, 0)
        scaler = torch.cuda.amp.GradScaler()
        for it in range(opt.num_itr):
            # -------- sample x_idx and t_idx \in [0, interval] --------
            if it!=0 and it%opt.resample_itr==0:
                train_gs, train_xis, train_ts, train_labels = self.sample_train_data(opt)

            # -------- build sample --------
            
            if not opt.random_x_t:
                samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,),device='cpu')
                samp_t_idx = util.time_sample(opt.interval, net.direction, opt.train_bs_t)
                if opt.use_arange_t: samp_t_idx = util.time_arange(train_ts.shape[0], net.direction)
                
                ts          = train_ts[samp_t_idx].detach()
                gs          = train_gs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
                xis         = train_xis[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
                labels      = train_labels[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
                
                gs          = util.flatten_dim01(gs)
                xis         = util.flatten_dim01(xis)
                labels      = util.flatten_dim01(labels)
                ts          = ts.repeat(opt.train_bs_x)
            else:
                samp_x_idx = torch.randint(opt.samp_bs,  (opt.ssm_batch,), device = 'cpu')
                samp_t_idx = util.time_sample(opt.interval, net.direction, opt.ssm_batch)
                
                gs          = train_gs[samp_x_idx , samp_t_idx, ...].to(opt.device)
                xis         = train_xis[samp_x_idx, samp_t_idx, ...].to(opt.device)
                labels      = train_labels[samp_x_idx,samp_t_idx, ...].to(opt.device)
                ts          = train_ts[samp_t_idx].detach()
            optimizer.zero_grad(set_to_none=True)

            # -------- handle for batch_x and batch_t ---------
            assert gs.shape[0] == ts.shape[0]
            assert xis.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            
            if opt.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss,zs = compute_SSM_train(opt, labels,self.dyn, ts, gs,xis, net, return_z=True)
            else:
                loss,zs = compute_SSM_train(opt, labels,self.dyn, ts, gs,xis, net, return_z=True)
            assert not torch.isnan(loss)
                
            scaler.scale(loss).backward()

            if opt.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update()
            if sched is not None: 
                with warmup.dampening():
                    sched.step()

            # -------- logging --------
            self.log_train(opt, it, loss, optimizer, 'backward')
            
            if it == 0: continue
            self.evaluate(opt, it+1)

    @torch.no_grad()
    def evaluate(self, opt, stage):
        snapshot, ckpt = util.evaluate_stage(opt, stage, metrics=None)
        
        if snapshot:
            z           = freeze(self.net)
            if stage==0:
                gs, xis, _  = self.dyn.sample_traj(self.ts, None, save_traj=True)
                fn          = "{}".format('forward_stage{}'.format(stage))
                fig, ax     = util.plot_func(
                                            opt,\
                                            fn,\
                                            gs.detach().cpu().numpy(),\
                                            xis.detach().cpu().numpy(),\
                                            n_snapshot=5,\
                                            direction='forward')

                if self.writer is not None: self.writer.add_plot_image(step=stage, key='forward', fig=fig)
                plt.close()
                
            with self.ema.average_parameters():
                gs, xis, _  = self.dyn.sample_traj(self.ts, z, save_traj=True)
            fn          = "{}_stage{}".format(z.direction,stage)
            fig, ax     = util.plot_func(
                                        opt,\
                                        fn,\
                                        gs.detach().cpu().numpy(),\
                                        xis.detach().cpu().numpy(),\
                                        n_snapshot=5,\
                                        direction=z.direction)
            
            plt.close()
            with self.ema.average_parameters():
                NLL  = compute_NLL(opt,z,self.dyn,self.dists.eval_loader)
            print(util.magenta('NLL loss is {}'.format(NLL)))
            
            self.net = activate(self.net)
            fig.suptitle("stage{}-{}".format(stage, z.direction))
            
            if self.writer is not None: 
                self.writer.add_plot_image(step=stage, key='backward', fig=fig)
                self.log_tb(stage, NLL.detach(), 'NLL', 'backward')
            if ckpt:
                keys = ['net','optimizer','ema']
                util.save_checkpoint(opt, self, keys, stage)

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+it)),
                num_itr,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    def log_train(self, opt, it, loss, optimizer, direction):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        if (it) % opt.log_iter == 0:
            print("train_it {0}/{1} | lr:{2} | loss:{3} | time:{4}"
                .format(
                    util.cyan("{}".format(it)),
                    opt.num_itr,
                    util.yellow("{:.2e}".format(lr)),
                    util.red("{:+.4f}".format(loss.item())),
                    util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),))
            
        if self.logging:
            self.log_tb(it, loss.detach(), 'loss', direction)

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(key=os.path.join(tag,name), val=val, step=step)

