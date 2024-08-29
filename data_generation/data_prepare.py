import torch
import numpy as np
import os
import abc
from omegaconf import OmegaConf

from pathlib import Path


def build_data(cfg):
    dataset_option = \
    {"electron1d_u(n)": Electron1DU,
     "gaussian_so(n)": GaussianSO,
     "spinglass_u(n)": SpinGlassU,
     }
    return dataset_option[cfg.data](cfg)


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

class LieGroup(abc.ABC):
    @abc.abstractmethod
    def _sample_algebra(self, batch):
        '''
        Sample distribution on Lie algbera
        '''
        pass

    def _sample_group(self, batch):
        lie_algebra_element = self._sample_algebra(batch)
        lie_group_element = torch.matrix_exp(lie_algebra_element)
        return lie_group_element

    def sample(self, batch):
        cwd = os.getcwd()
        print("Current Working Directory:", cwd)
        datadir = os.path.join('dataset')
        os.makedirs(datadir, exist_ok=True)
        filename = self.filename
        datasetdir = os.path.join(datadir, filename)
        lie_group_samples = self._sample_group(batch).numpy()

        np.save(datasetdir, lie_group_samples)

class GaussianSO(LieGroup):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg.dim
        self.batch = cfg.batch
        self.T = cfg.time

        self.filename = self.__class__.__name__ + f"dim_{self.dim}_batch_{self.batch}"

        self.mean, self.var = cfg.gaussian.mean, cfg.gaussian.var

    def _sample_algebra(self, batch):
        # Lie algbra so(n) is nxn skew symmetric matrix with diagonal 0
        pre_sample = self.mean + self.var * torch.randn(size = (batch, self.dim, self.dim))
        sample = 1/2 * (pre_sample - torch.transpose(pre_sample, dim0 = 1, dim1 = 2))
        sample = sample.to(torch.float32)

        return sample
    
class SpinGlassU(LieGroup):
    def __init__(self, cfg = None):
        self.cfg = cfg
        self.spin_num = cfg.spinglass.spin_num
        self.dim = 2 ** self.spin_num
        self.T = cfg.time
        self.batch = cfg.batch

        self.filename = self.__class__.__name__ + f"dim_{self.dim}_batch_{self.batch}"
    
        
        self.gamma_mean = 2
        self.gamma_var = 0.5
        
        self.J_low = 1
        self.J_high = 3
        
        self.sig_z = np.array([[1, 0], [0, -1]])
        self.sig_x = np.array([[0, 1], [1, 0]])
    
    def _sample_algebra(self, batch):
        interact, field = self._prepare_operator()
        J_coef = np.random.uniform(low = self.J_low, high = self.J_high, size = (batch, self.spin_num)).reshape(-1, self.spin_num, 1, 1)
        g_coef = self.gamma_var * np.random.randn(batch, self.spin_num).reshape(-1, self.spin_num, 1, 1) + self.gamma_mean
        
        interact_weight = J_coef * interact
        field_weight = g_coef * field
        Hamiltonian = 1j * self.T * (np.sum(interact_weight, axis = 1) + np.sum(field_weight, axis = 1))

        samples = torch.from_numpy(Hamiltonian).to(torch.complex64)
        return samples
        
    
    def _prepare_operator(self):
        interact = []
        field = []
        # compute interaction term
        for i in range(self.spin_num-1):
            before_spin = np.eye(2 ** i)
            spin = np.kron(self.sig_z, self.sig_z)
            after_spin = np.eye(2 ** (self.spin_num - i - 2))
            spin_operator = np.kron(before_spin, spin)
            spin_operator = np.kron(spin_operator, after_spin)
            interact.append(spin_operator)

        inbetween = np.eye(2 ** (self.spin_num - 2))
        spin_operator = np.kron(self.sig_z, inbetween)
        spin_operator = np.kron(spin_operator, self.sig_z)
        interact.append(spin_operator)
        
        # compute field term
        for i in range(self.spin_num):
            before_spin = np.eye(2 ** i)
            spin = np.kron(self.sig_x, np.eye(2 ** (self.spin_num - i - 1)))
            spin_operator = np.kron(before_spin, spin)
            field.append(spin_operator)
        
        interact = np.array(interact).reshape(1, self.spin_num, self.dim, self.dim)
        field = np.array(field).reshape(1, self.spin_num, self.dim, self.dim)
        
        return interact, field
            

class Electron1DU(LieGroup):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg.dim
        self.batch = cfg.batch
        self.T = cfg.time

        self.filename = self.__class__.__name__ + f"dim_{self.dim}_batch_{self.batch}"

        self.angular_min = self.cfg.electron1D.angular_min
        self.angular_max = self.cfg.electron1D.angular_max
        self.center_mean = self.cfg.electron1D.center_mean
        self.center_var = self.cfg.electron1D.center_var

    def _sample_algebra(self, batch):
        x_span = np.linspace(0, 1, num = 1 + self.dim)[: self.dim]

        DFT_1D = np.zeros((self.dim, self.dim), dtype=np.cfloat)
        for k in range(self.dim):
            DFT_1D[k, :] = 1/np.sqrt(self.dim) * np.array([np.exp(-1j * 2 * np.pi * k/self.dim * n) for n in range(self.dim)])

        DFT_1D_dag = DFT_1D.T.conj()
        frequency = np.concatenate([np.arange(0, self.dim//2 + 1), np.arange(-self.dim//2 + 1, 0)])
        laplacian = np.diag(np.array([- 4 * np.pi ** 2 * (ii ** 2) for ii in frequency]))
        delta_h = DFT_1D_dag @ laplacian @ DFT_1D

        center = self.center_mean + self.center_var * np.random.randn(batch)
        angular = np.random.uniform(low = self.angular_min, high = self.angular_max, size = batch)
        angular = 1/2 * np.square(angular)

        batch_V_diag = x_span.reshape(-1, 1).T - center.reshape(-1,1)
        batch_V_diag = angular[:, None] * np.square(batch_V_diag)

        delta_h = torch.tensor(delta_h, dtype=torch.complex64)
        batch_V_diag = torch.tensor(batch_V_diag, dtype = torch.complex64)
        batch_V = matrix_diag(batch_V_diag)

        hamiltonian = delta_h - batch_V
        sample = 1j * self.T * hamiltonian
        sample = sample.to(torch.complex64)
        return sample


def main():
    config_dir = os.path.join('data_generation','config.yaml')
    config = OmegaConf.load(config_dir)
    datasampler = build_data(config)
    datasampler.sample(config.batch)

def test_so():
    data_dir = os.path.join(os.getcwd(), 'dataset','GaussianSOdim_16_batch_1000.npy')
    data = np.load(data_dir)
    data = torch.tensor(data, dtype = torch.float32)
    N = data.shape[0]

    error1 = torch.bmm(data, torch.transpose(data, 1 ,2)) - matrix_diag(torch.ones(size = (data.shape[0], data.shape[1])))
    error2 = torch.bmm(torch.transpose(data, 1 ,2), data) - matrix_diag(torch.ones(size = (data.shape[0], data.shape[1])))

    error1 = torch.sum(torch.square(error1)) / N
    error2 = torch.sum(torch.square(error2)) / N
    print(error1, error2)

def test_u():
    data_dir = os.path.join(os.getcwd(), 'dataset','Electron1DUdim_16_batch_1000.npy')
    data = np.load(data_dir)
    data = torch.tensor(data, dtype = torch.complex64)
    N = data.shape[0]

    error1 = torch.bmm(data, torch.transpose(data, 1 ,2).conj()) - matrix_diag(torch.ones(size = (data.shape[0], data.shape[1])))
    error2 = torch.bmm(torch.transpose(data, 1 ,2).conj(), data) - matrix_diag(torch.ones(size = (data.shape[0], data.shape[1])))

    error1 = torch.sum(torch.square(error1)) / N
    error2 = torch.sum(torch.square(error2)) / N
    print(error1, error2)


if __name__ == "__main__":
    main()
