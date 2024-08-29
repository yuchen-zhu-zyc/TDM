import numpy as np
import abc
from tqdm import tqdm
import torch

import util

def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, dist):
    print(util.magenta("build base sde..."))
    return SimpleSDE(opt,dist)


class BaseIntegrator(metaclass=abc.ABCMeta):
    def __init__(self, opt,dist):
        self.opt        = opt
        self.dist       = dist
    @abc.abstractmethod
    def forward_prop(self, g, xi,dw):
        raise NotImplementedError
    @abc.abstractmethod        
    def backward_prop(self,g, xi,score,dw):
        raise NotImplementedError

    def propagate(self,g,xi,dw,direction,ode,score=None):
        if direction=='forward':
            return self.forward_prop(g,xi,dw, score = score, ode=ode)
        else:
            return self.backward_prop(g,xi, score, dw, ode=ode)


class ExpIntegrator(BaseIntegrator):
    def __init__(self, opt,dist):
        super(ExpIntegrator, self).__init__(opt,dist)
        self.dt = opt.T/opt.interval
    def forward_prop(self, g, xi, dw, ode=None, score = None):
        opt = self.opt
        score   = score.reshape_as(xi)
        bs,n,dim,_\
                =g.shape
        dt      = self.dt
        var     = np.sqrt(1- np.exp(-2 * dt)) / np.sqrt(dt)

        if ode:
            xi = np.exp(-dt) * xi - (1 - np.exp(-dt)) * score
            xiM = matrixlize_xi(xi,dim, mode=opt.mode)
            exp_xi = torch.matrix_exp(dt * xiM)
            g = torch.matmul(g, exp_xi)
        else:
            xi      =  np.exp(-dt) * xi + var * dw
            xiM     = matrixlize_xi(xi ,dim, mode = opt.mode)
            exp_xi  = torch.matrix_exp(dt * xiM)
            g       = torch.matmul(g, exp_xi)
        
        return g,xi

    def backward_prop(self,g, xi, score, dw, ode=None):
        opt = self.opt
        score   = score.reshape_as(xi)
        bs,n,dim,_\
                =g.shape
        dt      =self.dt
        var     = np.sqrt(np.exp(2 * dt) - 1) / np.sqrt(dt)
        if ode:
            xi = np.exp(dt) * xi + (np.exp(dt) - 1) * score
            xiM = matrixlize_xi(xi,dim, mode=opt.mode)
            exp_xi = torch.matrix_exp(-dt * xiM)
            g = torch.matmul(g, exp_xi)
            
        else:
            xi      = np.exp(dt) * xi + var * dw + (np.exp(dt) - 1) * 2 * score
            xiM     = matrixlize_xi(xi,dim, mode=opt.mode)
            exp_xi  = torch.matrix_exp(-dt * xiM)
            g       = torch.matmul(g, exp_xi)
            
        return g, xi

class EulerIntegrator(BaseIntegrator):
    def __init__(self, opt,dist):
        super(EulerIntegrator, self).__init__(opt,dist)
        self.dt = opt.T/opt.interval
    def forward_prop(self, g, xi, dw, score = None):
        opt = self.opt
        bs,n,dim,_\
                =g.shape
        score   =0
        sign    = 1
        dt      = self.dt
        xiM     = matrixlize_xi(xi,dim, mode=opt.mode)
        g       = g + sign * g @ xiM * dt
        xi      = xi - sign * xi * dt + 2 * score * dt+ np.sqrt(2) * dw
        return g, xi

    def backward_prop(self,g, xi, score, dw):
        opt = self.opt
        score   = score[...,None]
        sign    = -1
        bs,n,dim,_\
                =g.shape
        dt      = self.dt
        xiM     = matrixlize_xi(xi,dim, mode=opt.mode)
        g       = g + sign * g @ xiM * dt
        xi      = xi - sign * xi * dt + 2 * score * dt+ np.sqrt(2) * dw
        return g, xi



class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, dist):
        self.opt        = opt
        self.dt         = opt.T/opt.interval
        self.dist       = dist
        # debug()
        self.integrator ={
                            'Exp': ExpIntegrator,
                            'Euler': EulerIntegrator
                        }.get(opt.Integrator)(opt,dist)

    @abc.abstractmethod
    def sigma(self, x, t):
        raise NotImplementedError

    def dw(self, x, dt):
        return torch.randn_like(x) * np.sqrt(dt)

    def propagate(self, t, g, xi, score, direction, dw, ode=False):
        g,xi = self.integrator.propagate(g,xi,dw,direction,score=score, ode=ode)
        return g,xi


    def get_sigxi(self,ts):
        return torch.sqrt(1-torch.exp(-2*ts))
    def get_sigg(self,ts):
        return torch.sqrt(2*ts+8/(1+torch.exp(ts))-4)
        
    def xi_sample_score(self,xi0,ts):
        '''
        xi0: bs, n, dim x (dim-1)/2
        xit:bs,  n, dim x (dim-1)/2
        ts: bs,1
        '''
        bs,n,dim=xi0.shape
        ts      =  ts[:,None,...].repeat(1,n,1) # bs,n,1
        epsxi   = torch.randn_like(xi0)
        sigxi   =self.get_sigxi(ts)
        xit     = torch.exp(-ts) * xi0 + sigxi * epsxi
        scorexit= -epsxi/sigxi
        return xit, scorexit


    def g_sample_score(self,g0,xi0,xit,ts):
        '''
        g0: bs, n, dim, dim
        xi: bs, n, dim x (dim-1)/2
        xit:bs, n, dim x (dim-1)/2
        ts: bs,1
        '''
        ts              = ts.double()
        g0              = g0.double()
        bs,n,dim,_      = g0.shape
        ts              = ts[:,None,...].repeat(1,n,1) # bs,n,1
        # debug()
        mug             = (1-torch.exp(-ts))/(1+torch.exp(-ts))*(xit+xi0) #[bs,1]
        sigg            = self.get_sigg(ts)
        epsg            = torch.randn_like(mug)
        logSample       = mug + sigg * epsg # log (g0inv gt)
        
        ## Normalize the logSample, mug to be in the range of 0 to 2*pi
        logSample       = mod2pi(logSample)
        mug             = mod2pi(mug)
        
        matrixlogSample = matrixlize_xi(logSample,dim, mode=self.opt.mode)
        Sample = torch.linalg.matrix_exp(matrixlogSample)

        gt = torch.matmul(g0, Sample)
        
        gt              = gt.reshape(bs,n,dim*dim) #bs,n, dim*dim
        k               = self.opt.k
        kk              = 2*k+1
        sigg            = sigg.reshape(bs,n,1,1).repeat(1,1,kk,1).reshape(bs*kk*n,1)
        pis             = (torch.arange(-k,k+1)*2*torch.pi).reshape(1,1,kk,1).repeat(bs,n,1,1)
        vec_gt          = logSample.reshape(bs,n,1,1).repeat(1,1,kk,1).reshape(bs*n*kk,-1)
        mug             = mug.reshape(bs,n,1,-1).repeat(1,1,kk,1)
        mug             = (mug+pis).reshape(bs*n*kk,-1)

        pdfs            = util.gaussian_pdf(vec_gt, mug, sigg)
        pdfsum          = pdfs.reshape(bs,n,kk,1).sum(2)
        _ts             = ts.reshape(bs,n,1,1).repeat(1,1,kk,1).reshape(bs*kk*n,1)
        
        MultTerm        = 1/(-2*sigg**2)*2*(1-torch.exp(-_ts))/(1+torch.exp(-_ts))*(mug-vec_gt)
        nominator       = (pdfs*MultTerm).reshape(bs,n,kk,-1).sum(2) #bs,dim
        score           = nominator/pdfsum
        return gt.float(), score.float()

    @torch.no_grad()
    def dsm_sample(self, x0,ts):
        #xit =e^{-t}\xi0+sqrt{1-e^{-2t}}\eps
        #score_xi=-(xit-\muxi xi0)/(sigxi)=-(muxi*xi0+sigxi\eps-muxi*xi0)/(sig xi)=-eps
        g0,xi0          = x0['g'].to(self.opt.device),x0['xi'].to(self.opt.device)
        bs              = xi0.shape[0]
        xit,scorexit    = self.xi_sample_score(xi0,ts)
        gt,scoregt      = self.g_sample_score(g0,xi0,xit,ts)
        score           = scorexit+scoregt
        # debug()
        return gt,xit,score.reshape(bs,-1)
        
    @torch.no_grad()
    def sample_traj(self, ts, net, save_traj=True):
        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = net.direction if net is not None else 'forward'
        assert direction in ['forward', 'backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist_sample_func = self.dist.sample_data if direction == 'forward' else self.dist.sample_prior
        ts = ts if direction == 'forward' else torch.flip(ts, dims=[0])

        x       = init_dist_sample_func()  # [bs, x_dim]
        g, xi   = torch.Tensor(x['g']).to(ts.device), torch.Tensor(x['xi']).to(ts.device)
        bs      = g.shape[0]
        g_shape = g.shape[1:]


        gs      = torch.empty((bs, len(ts), *g.shape[1:]),device='cpu') if save_traj else None
        xis     = torch.empty((bs, len(ts), *xi.shape[1:]),device='cpu') if save_traj else None
        labels  = torch.empty_like(xis,device='cpu')
        
        if opt.mode == 'u':
            gs = gs.type(torch.complex64)

        _ts     = tqdm(ts, desc=util.yellow("Propagating Dynamics..."))
        
        for idx, t in enumerate(_ts):
            if direction == 'backward':
                score = net(vectorlize_g(g, mode=opt.mode), xi, t)
            else:
                score = torch.zeros_like(xi)

            t_idx   = idx if direction == 'forward' else len(ts)-idx-1
            dw      = self.dw(xi, self.dt)
            g, xi   = self.propagate(t, g, xi, score, direction, dw)
            
            if save_traj:
                # gs[:,t_idx,...]     = vectorlize_g(g, mode = opt.mode).detach().cpu()
                gs[:,t_idx,...]     = g.reshape(bs,*g_shape).detach().cpu()
                xis[:,t_idx,...]    = xi.detach().cpu()
                labels[:,t_idx,...] = (self.sigma(t)*(-dw)/np.sqrt(2)).detach().cpu()
        res = [gs, xis, labels]
        return res


class SimpleSDE(BaseSDE):
    def __init__(self, opt, dist, var=np.sqrt(2)):
        super(SimpleSDE, self).__init__(opt, dist = dist)
        self.var = var

    def sigma(self, t):
        return self.var*torch.ones_like(t)

def matrixlize_xi(xi, dim, mode='so'):
    assert mode in ['so', 'u']
    
    bs, num = xi.shape[0], xi.shape[1]
    upper_tri_indices = torch.triu_indices(dim, dim, offset=1)
    
    if mode == 'so':
        skew_matrices = torch.zeros(bs, num, dim, dim, dtype=xi.dtype, device=xi.device)
        # Make the matrix skew-symmetric by negating the transpose for the lower triangular part
        skew_matrices[:, :, upper_tri_indices[0], upper_tri_indices[1]] = xi
        skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)
    
    elif mode == 'u':
        skew_matrices = torch.zeros(bs, num, dim, dim, dtype=torch.complex64, device=xi.device)
        
        upper_tri_dim = int(dim * (dim - 1)// 2)
        skew_matrices[:, :,  upper_tri_indices[0], upper_tri_indices[1]] = xi[:, :, :upper_tri_dim] + 1j * xi[:, :, upper_tri_dim: 2 * upper_tri_dim]
        # Make the matrix skew-Hermite by negating the transpose conjugate for the lower triangular part
        skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1).conj()
        # Fill the diagonal with pure imaginary values
        skew_matrices = torch.diagonal_scatter(skew_matrices, src = 1j * xi[:, :, 2 * upper_tri_dim:], dim1 = -2, dim2=-1)
        
    return skew_matrices

def matrixlize_g(g, dim, mode='so'):
    assert mode in ['so', 'u']
    bs, num = g.shape[0], g.shape[1]
    if mode == 'so':
        g = g.reshape(bs, num, dim, dim)
        
    elif mode == 'u':
        g_real, g_imag = g[:, :, :int(dim * dim)], g[:, :, int(dim * dim):]
        g_real, g_imag = g_real.reshape(bs, num, dim, dim), g_imag.reshape(bs, num, dim, dim)
        g = g_real + 1j * g_imag
        
    return g

def vectorlize_g(g, mode = 'so'):
    assert mode in ['so', 'u']
    if mode == 'so':
        return g.reshape(g.shape[0], -1)
    
    elif mode == 'u':
        if len(g.shape) > 2:
            g_real, g_imag = g.real.reshape(g.shape[0], -1), g.imag.reshape(g.shape[0], -1)
            return torch.cat([g_real, g_imag], dim = -1)
        else:
            return g


def flat_dim01(x):
    return x.reshape(-1,*x.shape[2:])
def unflat_dim01(x,dim0,dim1):
    return x.reshape(dim0,dim1,*x.shape[1:])

def flat_dim012(x):
    return x.reshape(-1,*x.shape[3:])
def unflat_dim012(x,dim0,dim1,dim2):
    return x.reshape(dim0,dim1,dim2,*x.shape[1:])

def mod2pi(x):
    return torch.arctan2(torch.sin(x),torch.cos(x))