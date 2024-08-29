import numpy as np
import os
import argparse
import random
import torch

import configs
import util
from pathlib import Path

from ipdb import set_trace as debug


def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")

    # --------------- SDE ---------------
    parser.add_argument("--t0",             type=float, default=1e-2,     help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.,       help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--backward-net",   type=str,   default='toy',    choices=['toy'])
    parser.add_argument("--Integrator",     type=str,   default='Exp')
    
    # --------------- Neural Net ---------------
    parser.add_argument("--model-hidden-dim",  type=int,   default=256,  help="Model Hidden Dim")
    parser.add_argument("--model-blocks",   type=int,   default=7,       help="Model Residual Blocks")
    
    # --------------- training & sampling ---------------
    parser.add_argument("--status",         type=str,   default='train', choices=['eval','train'])
    parser.add_argument("--loss",           type=str,   default='ssm')
    parser.add_argument("--mixed-precision", action="store_true",         help="[Train] whether to use mixed precision FP16 for acceleration")
    parser.add_argument("--num-itr",        type=int,default=10000,       help="[Train] number of training iterations (for each epoch)")
    
    parser.add_argument("--resample-itr",   type=int,default=200,         help="[ISM training only] frequency of resampling the trajectory")
    parser.add_argument("--use-arange-t",   action="store_true",          help="[ISM training only] use full timesteps for training")
    parser.add_argument("--random-x-t",     action="store_true",          help="[ISM training only] use random x and t for training")
    parser.add_argument("--k",              type=int,   default=2,        help="[DSM training only] num of terms used in computation of DSM")

    parser.add_argument("--train-bs-x",     type=int,                     help="[ISM/DSM] batch size for training")
    parser.add_argument("--train-bs-t",     type=int,                     help="[ISM/DSM] batch size for timestep in training")
    
    parser.add_argument("--eval",           action="store_true",          help="[Eval] Evaluation mode")
    parser.add_argument("--eval-itr",       type=int, default=200,        help="[Eval] frequency of evaluation")
    parser.add_argument("--samp-bs",        type=int,                     help="[Eval] sampling batch size")


    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float,                   help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    
    
    # --------------- Problems and Lie group ---------------
    parser.add_argument("--problem-name",   type=str)
    parser.add_argument("--SON",            type=int,   default=3,        help="SOn dim")
    parser.add_argument("--UN",             type=int,   default=4,         help="Un dim") 
    parser.add_argument("--TORUS",          type=int,   default=1,        help="Torus dim") 
    parser.add_argument("--checker-board-pattern-num",  type=int,   default=4,       help="Checkerboard pattern")
    parser.add_argument("--spin-num",       type=int,   default=3,             help="Spin Glass num")
    parser.add_argument("--Protein-name",  type=str,   default= None,        help="Protein class name")
    
    # ---------------- evaluation ----------------
    parser.add_argument("--snapshot-freq",  type=int,   default=0,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=0,        help="checkpoint saving frequency w.r.t stages")
    
    # ---------------- Wandb ----------------
    parser.add_argument("--wandb-api-key",  type=str,   default= None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default= None,        help="user name of your W&B account")
    parser.add_argument("--wandb-project",  type=str,   default= None,        help="wandb project name")
    parser.add_argument("--name",           type=str,   default='debug',      help="wandb name, also saving dir")
    parser.add_argument("--log-dir",        type=Path,  default=".log",       help="path to log std outputs and writer data")
    

    problem_name = parser.parse_args().problem_name
    default_config, model_configs = {
        'SOn':              configs.get_SOn_default_configs,
        'toy':              configs.get_SOn_default_configs,
        'Protein':          configs.get_Tn_default_configs,
        'RNA':              configs.get_Tn_default_configs,
        'Un':               configs.get_Un_default_configs,
        'Checkerboard':     configs.get_Tn_default_configs,
        'GmmEulerAngle':    configs.get_SOn_default_configs,
        'Pacman':           configs.get_Tn_default_configs,
        'HighdimTorus':     configs.get_Tn_default_configs,
        'GmmAlgebra':       configs.get_SOn_default_configs,
        'SpinGlass':        configs.get_Un_default_configs,
    }.get(problem_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    
    # ========= auto setup & path handle =========
    opt.device='cuda:'+str(opt.gpu)
    torch.set_default_device(device = opt.device)
    opt.model_configs = model_configs


    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    if opt.snapshot_freq:
        opt.eval_path = os.path.join('results', opt.dir)
        os.makedirs(os.path.join(opt.eval_path), exist_ok=True)
        os.makedirs(os.path.join(opt.eval_path), exist_ok=True)

    opt.ckpt_path = os.path.join(opt.eval_path,'checkpoints')
    os.makedirs(os.path.join(opt.ckpt_path), exist_ok=True)

    # ========= print options =========
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)))
    print()

    return opt
