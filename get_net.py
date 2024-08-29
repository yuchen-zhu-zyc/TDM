
import torch
import util
import numpy as np
from ipdb import set_trace as debug
from sde import vectorlize_g

def build(opt, dyn, direction):
    print(util.magenta("build {} network...".format(direction)))

    net_name    = getattr(opt, direction+'_net')
    # debug()
    net         = _build_net(opt, net_name)
    norm_out    = (net_name in ['toy'])

    net = NetworkWrapper(
        opt, direction, dyn, net, norm_out=norm_out
    )
    print(util.red('number of parameters is {}'.format(util.count_parameters(net))))
    net.to(opt.device)

    return net

def _build_net(opt, net_name):
    if net_name == 'toy':
        assert util.is_toy_dataset(opt)
        from models.toy_model.Toy import build_toy, build_toy2
        net = build_toy2(opt, opt.data_dim, diag=opt.loss=='dsm', mode = opt.mode, dt = opt.dt, t0 = opt.t0)
    else:
        raise RuntimeError()
    return net

class NetworkWrapper(torch.nn.Module):
    # note: norm_out matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, norm_out=True):
        super(NetworkWrapper,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.norm_out = norm_out

    @ property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer

    def forward(self, x,v, t):
        dyn=self.dyn
        opt=self.opt
        # make sure t.shape = [batch]
        bs= x.shape[0]
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]
        
        if opt.mode == 'so':
            x  = x.reshape(bs,-1)
        elif opt.mode == 'u':
            x = vectorlize_g(x, mode=opt.mode)
            
        v  = v.reshape(bs,-1)

        out = self.net(x,v, t)
        if self.norm_out and self.opt.loss=='ssm':
            t       = t.reshape(out.shape[0], *([1,]*(out.dim()-1)))
            norm    = self.dyn.sigma(t)*np.sqrt(self.opt.T/self.opt.interval)/np.sqrt(2)
            out     = out * norm
        elif self.norm_out and self.opt.loss=='dsm':
            t       = t.reshape(out.shape[0], *([1,]*(out.dim()-1)))
            norm    = dyn.get_sigxi(t)
            out     = out/norm
        else:
            out=out

        return out


