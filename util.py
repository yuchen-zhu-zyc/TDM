import os, re

import numpy as np
import termcolor
import matplotlib.pyplot as plt
import torch

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from ipdb import set_trace as debug


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_image_dataset(opt):
    return opt.problem_name in ['mnist','cifar10','celebA32','celebA64']

def is_toy_dataset(opt):
    return True


def evaluate_stage(opt, stage, metrics):
    """ Determine what metrics to evaluate for the current stage,
    if metrics is None, use the frequency in opt to decide it.
    """
    if metrics is not None:
        return [k in metrics for k in ['snapshot', 'ckpt']]
    match = lambda freq: (freq>0 and stage%freq==0)
    return [match(opt.snapshot_freq), match(opt.ckpt_freq)]

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def compute_z_norm(zs, dt):
    zs = zs.reshape(*zs.shape[:2],-1)
    return 0.5 * zs.norm(dim=2).sum(dim=1).mean(dim=0) * dt

def create_traj_sampler(trajs):
    for traj in trajs:
        yield traj

def get_load_it(load_name):
    nums = re.findall('[0-9]+', load_name)
    assert len(nums)>0
    if 'stage' in load_name and 'dsm' in load_name:
        return int(nums[-2])
    return int(nums[-1])

def restore_checkpoint(opt, runner, load_name):
    assert load_name is not None
    print(green("#loading checkpoint {}...".format(load_name)))
    full_keys = ['net','optimizer','ema']

    with torch.cuda.device(opt.gpu):
        checkpoint = torch.load(load_name,map_location=opt.device)
        ckpt_keys=[*checkpoint.keys()]
        for k in ckpt_keys:
            getattr(runner,k).load_state_dict(checkpoint[k])

    if len(full_keys)!=len(ckpt_keys):
        value = { k for k in set(full_keys) - set(ckpt_keys) }
        print(green("#warning: does not load model for {}, check is it correct".format(value)))
    else:
        print(green('#successfully loaded all the modules'))

    runner.ema.copy_to()
    print(green('#loading form ema shadow parameter for neural network'))
    print(magenta("#######summary of checkpoint##########"))

def save_checkpoint(opt, runner, keys, stage_it, dsm_train_it=None):
    checkpoint = {}
    fn = opt.ckpt_path + "/stage_{0}{1}.npz".format(
        stage_it, '_dsm{}'.format(dsm_train_it) if dsm_train_it is not None else ''
    )
    with torch.cuda.device(opt.gpu):
        for k in keys:
            checkpoint[k] = getattr(runner,k).state_dict()
        torch.save(checkpoint, fn)
    print(green("checkpoint saved: {}".format(fn)))

def get_theta(gs):
    bs=gs.shape[0]
    gs=gs.reshape(bs,-1)
    inv_theta=np.arccos(gs[:,0,None])
    sign_sin = gs[:,1,None]
    inv_theta[sign_sin<0]=-inv_theta[sign_sin<0]
    return inv_theta

def share_axes(axes, sharex=True, sharey=True):
    if isinstance(axes, np.ndarray):
        axes = axes.flat  
    elif isinstance(axes, dict):
        axes = list(axes.values()) 
    else:
        axes = list(axes)
    ax0 = axes[-1]
    for ax in axes:
        if sharex:
            ax.sharex(ax0)
            if not ax.get_subplotspec().is_last_row():
                ax.tick_params(labelbottom=False)
        if sharey:
            ax.sharey(ax0)
            if not ax.get_subplotspec().is_first_col():
                ax.tick_params(labelleft=False)


def plot_func(opt,fn,gs,xis,n_snapshot,direction):
    dim = xis.shape[-1]
    if opt.problem_name in ['Checkerboard', 'Pacman']:
        return plot_checkerboard(opt,fn,gs,xis,n_snapshot,direction)
    elif opt.problem_name in ['GmmEulerAngle']:
        return plot_euleranlge(opt,fn,gs,xis,n_snapshot,direction)
    if dim ==1:
        return plot_traj_1d(opt,fn,gs,xis,n_snapshot,direction)
    else:
        return plot_traj(opt,fn,gs,xis,n_snapshot,direction)

def plot_dist_1d(opt,fn,gs,gs_gt,n_snapshot,direction):
    bs,steps,n,dim,_=gs.shape
    if opt.loss=='dsm':
        gs =gs[0:1000,0].reshape(1000*n,-1)
        gs_gt=gs_gt[0:1000].reshape(1000*n,-1)
    else:
        gs =gs[0:1000]
        xis =xis[0:1000]
    A=  (get_theta(gs[:,...])).reshape(1000,n)
    B=  (get_theta(gs_gt[:,...])).reshape(1000,n)
    fn_pdf = os.path.join('results', opt.dir, fn+'dist.pdf')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # figsize is adjustable

    axes[0, 0].hist(A[:, 0], bins=100, color='blue', alpha=0.7)
    axes[0, 0].set_title('Histogram of g0')

    axes[0, 1].hist(A[:, 1], bins=100, color='blue', alpha=0.7)
    axes[0, 1].set_title('Histogram of g0')

    axes[0, 2].hist(A[:, 2], bins=100, color='blue', alpha=0.7)
    axes[0, 2].set_title('Histogram of g0')

    axes[1, 0].hist(B[:, 0], bins=100, color='red', alpha=0.7)
    axes[1, 0].set_title('Histogram of g0_GT')

    axes[1, 1].hist(B[:, 1], bins=100, color='red', alpha=0.7)
    axes[1, 1].set_title('Histogram of g0_GT')

    axes[1, 2].hist(B[:, 2], bins=100, color='red', alpha=0.7)
    axes[1, 2].set_title('Histogram of g0_GT')

    plt.tight_layout()
    plt.savefig(fn_pdf)
    return fig, _

def plot_traj_1d(opt,fn,gs,xis,n_snapshot,direction):
    if opt.loss=='dsm':
        gs =gs[0:1000,:,0]
        xis =xis[0:1000,:,0]
    else:
        gs =gs[0:1000]
        xis =xis[0:1000]
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    bs = gs.shape[0]
    gs  = gs
    xis  = xis
    fig, axss = plt.subplots(2, n_snapshot, figsize=(10, 4))
    total_steps = gs.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)

    color = 'salmon' if direction=='forward' else 'royalblue'
    axs=axss[0]
    for ax, step in zip(axs, sample_steps):
        theta=get_theta(gs[:,step,...])
        ax.hist(theta, bins=100,color=color)
        ax.set_xticks(np.arange(-3, 3, 1))
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        
    axs=axss[1]
    for ax, step in zip(axs, sample_steps):
        ax.hist(xis[:,step], bins=100,color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
    
    fig.tight_layout()
    plt.savefig(fn_pdf)
    return fig, ax

def plot_traj(opt,fn,gs,xis,n_snapshot,direction):
    gs =gs[0:1000,:,0,0]
    xis =xis[0:1000,:,0,:]
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    bs = gs.shape[0]
    
    if np.iscomplexobj(gs):
        gs = gs.real
    # gs  = gs
    xis  = xis
    fig, axss = plt.subplots(2, n_snapshot, figsize=(10, 4))
    total_steps = gs.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)

    color = 'salmon' if direction=='forward' else 'royalblue'
    axs=axss[0]
    
    for ax, step in zip(axs, sample_steps):
        # theta=get_theta(gs[:,step,...])
        ax.scatter(gs[:,step,0],gs[:,step,2], s=1, color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
    axs=axss[1]
    for ax, step in zip(axs, sample_steps):
        ax.scatter(xis[:,step,0],xis[:,step,2], s=1, color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
    
    share_axes(axss[0, :])
    share_axes(axss[1, :])   
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fn_pdf)
    return fig, ax


def plot_checkerboard(opt,fn,gs,xis,n_snapshot,direction):
    assert opt.problem_name in ['Checkerboard', 'Pacman']
    gs = gs[:,:, :]
    xis = xis[:,:, :, 0]
    
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    bs = gs.shape[0]
    gs  = gs
    xis  = xis
    fig, axss = plt.subplots(2, n_snapshot, figsize=(10, 4))
    total_steps = gs.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
    color = 'salmon' if direction=='forward' else 'royalblue'
    axs=axss[0]
    for ax, step in zip(axs, sample_steps):
        theta1, theta2 = get_theta(gs[:,step,0, ...]), get_theta(gs[:,step, 1, ...])
        ax.scatter(theta1, theta2, s = 1, color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        
    axs=axss[1]
    for ax, step in zip(axs, sample_steps):
        ax.scatter(xis[:,step,0],xis[:,step,1], s=1, color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))    
    fig.tight_layout()
    plt.savefig(fn_pdf)
    return fig, ax

def plot_euleranlge(opt,fn,gs,xis,n_snapshot,direction):
    assert opt.problem_name == 'GmmEulerAngle'
    gs = gs[0:1000,:, 0, :]
    xis = xis[0:1000, :, 0]
    
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    bs = gs.shape[0]
    gs  = gs
    xis  = xis
    fig, axss = plt.subplots(4, n_snapshot, figsize=(10, 8))
    total_steps = gs.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
    color = 'salmon' if direction=='forward' else 'royalblue'
    axs0, axs1, axs2 = axss[0], axss[1], axss[2]
    for ax0, ax1, ax2, step in zip(axs0, axs1, axs2, sample_steps):
        euler_angle = batch_rotation_matrix_to_euler(gs[:, step, ...])
        ax0.hist(euler_angle[:, 0], bins=100, color=color)
        ax1.hist(-euler_angle[:, 1], bins=100, color=color)
        ax2.hist(euler_angle[:, 2], bins=100, color=color)
        ax0.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        ax1.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        ax2.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        ax0.set_xlim([-np.pi, np.pi])
        ax1.set_xlim([-np.pi/2, np.pi/2])
        ax2.set_xlim([-np.pi, np.pi])     
    axs=axss[3]
    for ax, step in zip(axs, sample_steps):
        ax.scatter(xis[:,step,0],xis[:,step,2], s=1, color=color)
        ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))    
        
    fig.tight_layout()
    plt.savefig(fn_pdf)
    return fig, ax



def time_sample(_interval, direction,num_samp):
    if direction=='backward':
        return torch.randint(1,_interval,(num_samp,)).cpu()
    else:
        return torch.randint(0,_interval-1,(num_samp,)).cpu()
    
def time_arange(_interval, direction):
    if direction=='backward':
        return torch.arange(1,_interval).cpu()
    else:
        return torch.arange(0,_interval-1).cpu()

def uniform(bs,r1,r2,device):
    return (r1 - r2) * torch.rand(bs,device=device) + r2

def log_density_multivariate_normal(x):
    k = x.shape[1]  # Dimension of the Gaussian
    constant = -0.5 * k * np.log(2 * torch.pi)
    log_density = constant - 0.5 * torch.sum(x**2, dim=1)
    return log_density

def gaussian_pdf(x, mu, sigma):
    normalization = 1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
    exponent = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    pdf = normalization * exponent
    
    return pdf


def torch_logm(A):
    lam, V = torch.linalg.eig(A)
    V_inv = torch.inverse(V).to(torch.complex128)
    V = V.to(torch.complex128)
    
    if not torch.allclose(A.to(torch.complex128), V @ torch.diag(lam).to(torch.complex128) @ V_inv):
        raise ValueError("Matrix is not diagonalizable, cannot compute matrix logarithm!")
    
    log_A_prime = torch.diag(lam.log())
    return V @ log_A_prime @ V_inv


def batch_euler_to_rotation_matrix(roll, pitch, yaw):
    # Ensure that roll, pitch, and yaw are numpy arrays
    roll = np.array(roll) # [-pi, pi]
    pitch = np.array(pitch) # [-pi/2, pi/2]
    yaw = np.array(yaw)  # [-pi, pi]

    # Number of matrices to compute
    batch_size = roll.shape[0]

    # Precompute cosines and sines of the angles
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # Initialize an empty array for the rotation matrices
    rotation_matrices = np.zeros((batch_size, 3, 3))

    # Populate the rotation matrices
    rotation_matrices[:, 0, 0] = cos_p * cos_y
    rotation_matrices[:, 0, 1] = sin_r * sin_p * cos_y - cos_r * sin_y
    rotation_matrices[:, 0, 2] = cos_r * sin_p * cos_y + sin_r * sin_y
    rotation_matrices[:, 1, 0] = cos_p * sin_y
    rotation_matrices[:, 1, 1] = sin_r * sin_p * sin_y + cos_r * cos_y
    rotation_matrices[:, 1, 2] = cos_r * sin_p * sin_y - sin_r * cos_y
    rotation_matrices[:, 2, 0] = -sin_p
    rotation_matrices[:, 2, 1] = sin_r * cos_p
    rotation_matrices[:, 2, 2] = cos_r * cos_p

    return rotation_matrices

def batch_rotation_matrix_to_euler(R):
    # Ensure R is a numpy array with shape (batch_size, 3, 3)
    assert R.shape[1:] == (3, 3), "Each item in the batch must be a 3x3 matrix"

    # Allocate space for the Euler angles: one row per matrix
    batch_size = R.shape[0]
    euler_angles = np.zeros((batch_size, 3))

    # Sy calculation for singular check
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)

    # Singular and non-singular indices
    singular_mask = sy < 1e-6
    non_singular_mask = ~singular_mask

    # Non-singular case calculations
    euler_angles[non_singular_mask, 0] = np.arctan2(R[non_singular_mask, 2, 1], R[non_singular_mask, 2, 2])
    euler_angles[non_singular_mask, 1] = np.arctan2(-R[non_singular_mask, 2, 0], sy[non_singular_mask])
    euler_angles[non_singular_mask, 2] = np.arctan2(R[non_singular_mask, 1, 0], R[non_singular_mask, 0, 0])

    # Singular case calculations
    euler_angles[singular_mask, 0] = np.arctan2(-R[singular_mask, 1, 2], R[singular_mask, 1, 1])
    euler_angles[singular_mask, 1] = np.arctan2(-R[singular_mask, 2, 0], sy[singular_mask])
    euler_angles[singular_mask, 2] = 0

    return euler_angles