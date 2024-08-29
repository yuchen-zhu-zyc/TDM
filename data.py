import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import abc

import util
import pandas as pd
from ipdb import set_trace as debug

def setup_loader(data_eval,batch):
    return DataLoader(\
                    data_eval,\
                    batch_size=batch,\
                    shuffle=True,
                    generator=torch.Generator(device='cuda'),
                    )

def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))
    if opt.problem_name == 'toy':
        data        = toyData()
        opt.data_dim= [data.X_train.shape[1],data.X_train.shape[2],data.X_train.shape[3]]
    elif opt.problem_name == 'Protein':
        data = ProteinData(name=opt.Protein_name)
        opt.data_dim = [2,2,2]
    elif opt.problem_name == 'RNA':
        data = RnaData()
        opt.data_dim = [7,2,2]
    elif opt.problem_name == 'SOn':
        data_dim = opt.SON
        data_dir = 'dataset/GaussianSOdim_{}_batch_100000.npy'.format(data_dim)
        data        = SOData(data_dir, data_dim, opt.samp_bs)
        opt.data_dim= [1,opt.SON,opt.SON]
    elif opt.problem_name == 'Un':
        data_dim = opt.UN
        data_dir = 'dataset/Electron1DUdim_{}_batch_100000.npy'.format(data_dim)
        data        = UData(data_dir, data_dim, opt.samp_bs)
        opt.data_dim= [1,opt.UN,opt.UN]
    elif opt.problem_name == "SpinGlass":
        data_dim = 2 ** opt.spin_num
        data_dir     = 'dataset/SpinGlassUdim_{}_batch_100000.npy'.format(data_dim)
        data         = UData(data_dir, data_dim, opt.samp_bs)
        opt.data_dim = [1, data_dim, data_dim]  
    elif opt.problem_name == 'Checkerboard':
        data = CheckerboardData(pattern_num=opt.checker_board_pattern_num)
        opt.data_dim = [2,2,2]
    elif opt.problem_name == 'Pacman':
        data = Pacman(batch_size=opt.samp_bs)
        opt.data_dim = [2,2,2]    
    elif opt.problem_name == 'GmmEulerAngle':
        data = GmmEulerAngle(opt.samp_bs)    
        opt.data_dim = [1,3,3]
    elif opt.problem_name == 'GmmAlgebra':
        data_dir = "dataset/rsgm_SO3_32_batch_100000.npy"
        data = GmmAlgebra(data_dir, opt.samp_bs)    
        opt.data_dim = [1, 3, 3]
        
    elif opt.problem_name == "HighdimTorus":
        data = HighdimTorus(batch_size=opt.samp_bs, dim = opt.TORUS)
        opt.data_dim = [opt.TORUS, 2, 2]
    else:
        raise RuntimeError()
    return data

class LieGroupPrior(abc.ABC):
    @abc.abstractmethod
    def sample_Haar(self, batch):
        pass
    @abc.abstractmethod
    def sample_Normal(self, batch):
        pass

class LieData(abc.ABC):
    @abc.abstractmethod
    def sample_prior(self):
        pass
    @abc.abstractmethod
    def sample_data(self):
        pass
    
    def sample_prior(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        m['g'] = self.prior.sample_Haar(batch_size)
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m

    def sample_data(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        index = torch.randint(0, len(self.data_train), (batch_size,), device='cpu')
        
        m['g'] = self.data_train[index]
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m
    


class SO_Prior(LieGroupPrior):
    def __init__(self, dim, num = 1):
        self.dim = dim
        self.num = num

    def sample_Haar(self, batch):
        flip_mat = torch.diag(torch.tensor([-1] + [1] * (self.dim - 1))).to(torch.float32)
        ON = self._sample_ON(batch * self.num)
        qualified_mask = torch.sign(torch.det(ON)) == 1
        SON = ON[qualified_mask]
        ON = ON[~qualified_mask]
        SON_2 = ON @ flip_mat
        SON = torch.concatenate([SON, SON_2]).to(torch.float32)
        SON = SON.reshape(batch, self.num, self.dim, self.dim)
        return SON
    
    def _sample_ON(self, batch):
        A = torch.randn(batch, self.dim, self.dim)
        Q, R = torch.linalg.qr(A)
        factor = torch.diagonal(R, dim1=1, dim2=2)
        factor = factor / torch.abs(factor)
        factor = torch.diag_embed(factor)
        Q = torch.bmm(Q, factor)
        return Q

    def sample_Normal(self, batch):
        algebra_dim = int((self.dim ** 2 - self.dim)/2)
        return torch.randn(batch, self.num, algebra_dim)

class U_Prior(LieGroupPrior):
    def __init__(self, dim, num = 1):
        self.dim = dim
        self.num = num

    def sample_Haar(self, batch):
        A = 1/np.sqrt(2) * (torch.randn(batch * self.num,self.dim, self.dim) +
                            1j * torch.randn(batch * self.num, self.dim, self.dim))
        Q, R = torch.linalg.qr(A)
        factor = torch.diagonal(R, dim1=1, dim2=2)
        factor = factor / torch.abs(factor)
        factor = torch.diag_embed(factor)
        Q = torch.bmm(Q, factor)
        Q = Q.reshape(batch, self.num, self.dim, self.dim)
        return Q

    def sample_Normal(self, batch):
        algebra_dim = int(self.dim ** 2)
        return torch.randn(batch, self.num, algebra_dim)


def uniform(r1, r2, a, b):
    return (r1 - r2) * torch.rand(a, b) + r2


def batchlize_matrix(bs, x):
    x = x[None, ...]
    x = x.repeat(bs, 1, 1)
    return x

class toyData(LieData):
    def __init__(self, dim=2, batch_size=5000):
        self.dim        = dim
        self.batch_size = batch_size
        self.prior      = SO_Prior(dim)
        num_samples     = 10000
        m               = {}
        
        Thetas          =[]
        for theta_range in [1/6, 1/4]:
            tmp=[uniform(r,r+theta_range*np.pi,int(num_samples/4),1) for r in [0,1/2*np.pi, np.pi, 3/2*np.pi]]
            Thetas.append(tmp)
        data=[]
        self.power      = len(Thetas)
        for theta in Thetas:
            init_gs=[]
            for theta_0 in theta:
                diag_term   = torch.ones(dim)*torch.cos(theta_0)
                init_g      = torch.diag_embed(diag_term)
                init_g[:,0,1] = torch.sin(theta_0.squeeze())
                init_g[:,1,0] = -torch.sin(theta_0.squeeze())
                init_gs.append(init_g)
            init_gs=torch.cat(init_gs,dim=0)
            init_gs=init_gs[torch.randperm(init_gs.size()[0])]
            data.append(init_gs[:,None,...])
        data    = torch.cat(data,dim=1) #bs,n,2,2
        data    = data.cpu().numpy()
        # debug
        X_train, X_eval = train_test_split(data, test_size=0.1)
        self.X_train    = torch.from_numpy(X_train)
        # debug()
        self.eval_loader=setup_loader(torch.from_numpy(X_eval),batch_size)


    def sample_prior(self, batch_size=None):
        batch_size  = batch_size if batch_size is not None else self.batch_size
        dim         = self.dim
        m           = {}
        m['g']      = (self.prior.sample_Haar(batch_size*self.power)).reshape(batch_size,self.power,dim,dim)
        m['xi']     = (self.prior.sample_Normal(batch_size*self.power)).reshape(batch_size,self.power,-1)
        return m

    def sample_data(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        index = torch.randint(0, len(self.X_train), (batch_size,), device='cpu')
        m['g'] = self.X_train[index]
        m['xi'] = (self.prior.sample_Normal(batch_size*self.power)).reshape(batch_size,self.power,-1)
        return m

class SOData(LieData):
    def __init__(self, data_dir=None, dim=3, batch_size=5000):
        self.dim        = dim
        self.batch_size = batch_size
        self.data_dir   = data_dir
        self.prior      = SO_Prior(dim)
        
        # Reshape to [bs, 1, dim, dim]
        data            = np.load(data_dir).reshape(-1, 1, dim, dim)
        X_train, X_eval = train_test_split(data, test_size=0.1)

        self.X_train    = torch.from_numpy(X_train)
        self.eval_loader= setup_loader(X_eval,batch_size)


    def sample_prior(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        m['g'] = self.prior.sample_Haar(batch_size)
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m

    def sample_data(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        index = torch.randint(0, len(self.X_train), (batch_size,), device='cpu')
        m['g'] = self.X_train[index]
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m
    
class GmmEulerAngle(LieData):
    def __init__(self, batch_size = 5000):
        self.dim = 3
        self.num = 1
        self.batch_size = batch_size
        
        roll_means = [0, -2, 2]  
        roll_variances = [0.2, 0.2, 0.2]
        roll_weights = [0.5, 0.2, 0.3]
        
        pitch_means = [0, 1]
        pitch_variances = [0.1, 0.3]
        pitch_weights = [0.3 , 0.7]
        
        yaw_means = [0, -3, 3]
        yaw_variances = [0.2, 0.2, 0.2]
        yaw_weights = [0.5, 0.2, 0.3]
        
        self.prior = SO_Prior(self.dim, self.num)
        
        roll = self.sample_gaussian_mixture(roll_means, roll_variances, roll_weights, 100000)
        pitch = self.sample_gaussian_mixture(pitch_means, pitch_variances, pitch_weights, 100000)
        yaw = self.sample_gaussian_mixture(yaw_means, yaw_variances, yaw_weights, 100000)
        
        roll = (roll + np.pi) % (2 * np.pi) - np.pi
        pitch = (pitch + np.pi/2) % np.pi - np.pi/2
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        
        self.data = util.batch_euler_to_rotation_matrix(roll, pitch, yaw).reshape(-1, 1, 3, 3)
        self.data = torch.from_numpy(self.data).float()
        
        X_train, X_eval = train_test_split(self.data, test_size=0.1)
        self.data_train = X_train
        self.eval_loader = setup_loader(X_eval, batch_size)
        
        
    def sample_gaussian_mixture(self, means, variances, weights, num_samples):
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= weights.sum()
        component_indices = np.random.choice(len(weights), size=num_samples, p=weights)
        # Sample from the appropriate Gaussian for each selected component
        samples = np.array([
            np.random.normal(means[i], np.sqrt(variances[i]))
            for i in component_indices
        ])

        return samples

class CheckerboardData(LieData):
    def __init__(self, pattern_num = 4, batch_size = 5000):
        self.dim = 2
        self.num = 2
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        data_train, data_eval = self._make_checkerboard(pattern_num)
        self.eval_loader= setup_loader(data_eval, batch_size)
        self.data_train = data_train

    def _make_checkerboard(self, pattern_num, batch_size = 200000):        
        x_coords = []
        y_coords = []
        n, points_per_square = pattern_num, batch_size // (pattern_num ** 2)
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    points_x = np.random.uniform(i, i + 1, points_per_square)
                    points_y = np.random.uniform(j, j + 1, points_per_square)
                    x_coords.extend(points_x)
                    y_coords.extend(points_y)
                    
        data = np.stack([x_coords, y_coords], axis=1)
        checkerboard = (data / n - 1/2) * 2 * np.pi
        data_arr = torch.from_numpy(checkerboard).float().reshape(-1, 2)
        data_mat = torch.stack([torch.cos(data_arr), torch.sin(data_arr), -torch.sin(data_arr), torch.cos(data_arr)], dim = -1).reshape(-1, 2, 2, 2)
        data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.05)
        return data_mat_train, data_mat_eval
    
class Pacman(LieData):
    def __init__(self, batch_size = 5000):
        self.dim = 2
        self.num = 2
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        data_arr = torch.from_numpy(np.load('dataset/pacman.npy')).float().reshape(-1,2)
        data_arr = (data_arr / 1000 - 1/2) * 2 * np.pi
        
        data_mat = torch.stack([torch.cos(data_arr), torch.sin(data_arr), -torch.sin(data_arr), torch.cos(data_arr)], dim = -1).reshape(-1, 2, 2, 2)
        data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.05)
        self.data_train = data_mat_train
        self.eval_loader = setup_loader(data_mat_eval, batch_size)
        
class HighdimTorus(LieData):
    def __init__(self, dim = 3, batch_size = 5000):
        self.dim = 2
        self.num = dim
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        data_mat = torch.from_numpy(np.load(f"dataset/HighdimTorus_{dim}_batch_100000.npy")).float()
        data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.05)
        self.data_train = data_mat_train
        self.eval_loader = setup_loader(data_mat_eval, batch_size)
        
class GmmAlgebra(LieData):
    def __init__(self, data_dir, batch_size = 5000):
        self.dim = 3
        self.num = 1
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        data_mat = torch.from_numpy(np.load(data_dir)).float()
        data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.1)
        self.data_train = data_mat_train
        self.eval_loader = setup_loader(data_mat_eval, batch_size)
        
class ProteinData(LieData):
    def __init__(self, batch_size = 5000,name=None):
        self.dim = 2
        self.num = 2
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        
        df = pd.read_csv('dataset/protein_angles.tsv', sep='\t', header = None)
        self.data_type  = ['General', 'Glycine', 'Pre-Pro', 'Proline']
        self.data_train, data_eval = self._preprocessing(df)
        self.data_train = self.data_train[name].float()
        self.data_eval       = data_eval[name].float()
        self.eval_loader= setup_loader(self.data_eval,batch_size)

    def _preprocessing(self, df):
        data_dict_train, data_dict_eval = {}, {}
        for name in self.data_type:
            data_df = df[df[3] == name][[1,2]]
            N = len(df)
            data_arr = data_df.to_numpy()
            data_arr = data_arr / 180 * np.pi
            data_arr = torch.from_numpy(data_arr).reshape(-1, 2)
            data_mat = torch.stack([torch.cos(data_arr), torch.sin(data_arr), -torch.sin(data_arr), torch.cos(data_arr)], dim = -1).reshape(-1, 2, 2, 2)
            
            data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.1)
            data_dict_train[name], data_dict_eval[name] = data_mat_train, data_mat_eval
        
        return data_dict_train, data_dict_eval
    
    def sample_prior(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        m['g'] = self.prior.sample_Haar(batch_size)
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m

    def sample_data(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        index = torch.randint(0, len(self.data_train), (batch_size,), device='cpu')
        
        m['g'] = self.data_train[index]
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m
    
class RnaData(LieData):
    def __init__(self, batch_size = 5000):
        self.dim = 2
        self.num = 7
        self.batch_size = batch_size
        self.prior = SO_Prior(self.dim, self.num)
        
        df = pd.read_csv('dataset/rna_angles.tsv', sep='\t', header = None)
        self.data_train, data_eval = self._preprocessing(df)
        self.data_train=self.data_train.float()
        data_eval=data_eval.float()
        self.eval_loader=setup_loader(data_eval,batch_size)
        
    def _preprocessing(self, df):
        data_df = df.drop([0,1], axis = 1)
        data_arr = data_df.to_numpy()
        data_arr = data_arr / 180 * np.pi
        data_arr = torch.from_numpy(data_arr).reshape(-1, 7)
        data_mat = torch.stack([torch.cos(data_arr), torch.sin(data_arr), -torch.sin(data_arr), torch.cos(data_arr)], dim = -1).reshape(-1, 7, 2, 2)
        data_mat_train, data_mat_eval = train_test_split(data_mat, test_size=0.1)
        return data_mat_train, data_mat_eval
    
    
    def sample_prior(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        m['g'] = self.prior.sample_Haar(batch_size)
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m

    def sample_data(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        m = {}
        index = torch.randint(0, len(self.data_train), (batch_size,), device='cpu')
        
        m['g'] = self.data_train[index]
        m['xi'] = self.prior.sample_Normal(batch_size)
        return m
    

class UData(LieData):
    def __init__(self, data_dir, dim, batch_size=5000):
        self.dim = dim
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prior = U_Prior(dim)
        
        # Reshape to [bs, 1, dim, dim]
        data = np.load(data_dir).reshape(-1, 1, dim, dim)
        
        X_train, X_eval = train_test_split(data, test_size=0.1)

        self.data_train    = torch.from_numpy(X_train)
        self.eval_loader = setup_loader(torch.from_numpy(X_eval),batch_size)
