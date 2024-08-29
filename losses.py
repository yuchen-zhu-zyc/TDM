import numpy as np
import torch
import util
from sde import vectorlize_g, matrixlize_xi
from tqdm import tqdm
from ipdb import set_trace as debug
from torch.func import vjp, jvp, vmap, jacrev

def compute_SSM_train(opt, label,dyn, ts, gs,xis, net, return_z=False):
    dt      = dyn.dt
    zs      = net(gs,xis,ts)
    label = label.reshape(label.shape[0],-1)
    g_ts    = dyn.sigma(ts)
    g_ts    = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    _ts     = ts.reshape(label.shape[0], *([1,]*(label.dim()-1)))
    reweight=1/(dyn.sigma(_ts)*np.sqrt(dt)/np.sqrt(2))
    
    loss    = torch.nn.functional.mse_loss(reweight*g_ts*dt*zs,reweight*label)
    return loss, zs if return_z else loss

def compute_DSM_train(reweight,pred,label):
    return torch.nn.functional.mse_loss(reweight*pred,reweight*label)


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, -div-x.shape[-1]

def compute_NLL(opt,net,dyn,eval_loader):
    n,dim,_=opt.data_dim
    loglik = 0
    num_data= eval_loader.dataset.shape[0]
    
    if opt.problem_name in ["Protein", "RNA", "Checkerboard", "Pacman", "HighdimTorus"]:
        offset  = n*np.log(2*np.pi)
    elif opt.problem_name in ["GmmAlgebra"]:
        offset = n * np.log(8 * np.pi ** 2)
    else:
        offset = 0
    for idx, g in enumerate(tqdm(eval_loader,desc=util.yellow("Evaluating NLL"))):
        g           = g.to(opt.device)
        bs,n,dim,_  = g.shape
        xi_dim      = dim ** 2  if opt.mode=='u' else int((dim*(dim-1))/2)
        xi          = torch.randn(bs,n,xi_dim)
        logp0xi     = util.log_density_multivariate_normal(xi.reshape(bs,-1))
        _ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        acc_Logprob = 0
        vv          = torch.randn_like(xi.reshape(bs,-1))
        for idx, t in enumerate(_ts):
            _t = t.repeat(bs)[...,None]
            vecfield = lambda z: net(vectorlize_g(g, mode = opt.mode),z, _t)
            score, div = output_and_div(vecfield, xi.reshape(bs,-1),v=vv,div_mode='approx')
            acc_Logprob = acc_Logprob+div*opt.dt
            g, xi       = dyn.propagate(t, g, xi, score, 'forward', dw=0,ode=True)
        logpTg  = 0
        logpTxi = util.log_density_multivariate_normal(xi.reshape(bs,-1))
        logp0g  = acc_Logprob+logpTg+logpTxi-logp0xi
        logp0g  = logp0g-offset # divide over dim
        loglik  = loglik+logp0g.sum() #sum the loglikelihood for all batch
    avg_loglik= loglik/num_data
    return -avg_loglik # average over batch