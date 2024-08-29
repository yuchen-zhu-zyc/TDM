import torch
import torch.nn as nn
from models.utils import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def build_toy(opt, dim, diag, mode='so'):
    assert mode in ['so', 'u']
    if diag:
        n_diag, dim, _ = dim
        input_dim = n_diag*2*2+n_diag
        out_dim = n_diag
    else:
        n_diag, dim, _ = dim
        if mode == 'so':
            input_dim = dim**2+int((dim*(dim-1))/2)
            out_dim = int((dim*(dim-1))/2)
        elif mode == 'u':
            input_dim = 2 * dim**2 + dim ** 2
            out_dim = int(dim ** 2)
    return ResNet(input_dim, out_dim, hidden_dim=opt.model_hidden_dim, n_hidden_layers=opt.model_layers)

def build_toy2(opt, dim, diag, dt, t0, mode='so'):
    assert mode in ['so', 'u']
    if diag:
        n_diag, dim, _ = dim
        input_dim = n_diag * dim**2 + n_diag * int((dim*(dim-1))/2)
        out_dim = n_diag * int((dim*(dim-1))/2)
    else:
        n_diag, dim, _ = dim
        if mode == 'so':
            input_dim = dim**2+int((dim*(dim-1))/2)
            out_dim = int((dim*(dim-1))/2)
        elif mode == 'u':
            input_dim = 2 * dim**2 + dim ** 2
            out_dim = int(dim ** 2)
    return ResNet2(input_dim, out_dim, dt = dt, t0 = t0, hidden_dim = opt.model_hidden_dim, num_res_blocks=opt.model_blocks)

class ResNet(nn.Module):
    def __init__(self,
                 input_dim=2,
                 out_dim=None,
                 index_dim=1,
                 hidden_dim=512,
                 n_hidden_layers=20):
        super().__init__()

        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers
        # in_dim = input_dim**2+int((input_dim*(input_dim-1))/2)

        # out_dim=int((input_dim*(input_dim-1))/2)

        layers = []
        layers.append(torch.jit.script(nn.Linear(input_dim+1, hidden_dim)))
        for _ in range(n_hidden_layers):
            layers.append(torch.jit.script(
                nn.Linear(hidden_dim + index_dim, hidden_dim)))
        layers.append(torch.jit.script(
            nn.Linear(hidden_dim + index_dim, out_dim)))

        self.layers = nn.ModuleList(layers)
        self.layers[-1] = zero_module(self.layers[-1])

    def _append_time(self, h, t):
        time_embedding = torch.log(t)
        return torch.cat([h, time_embedding.reshape(-1, 1)], dim=1)

    def forward(self, x, v, t):
        bs = x.shape[0]
        # try:
        # except:
        #     debug()
        u = torch.cat([x, v], dim=-1)
        h0 = self.layers[0](self._append_time(u, t))
        h = self.act(h0)

        for i in range(self.n_hidden_layers):
            h_new = self.layers[i + 1](self._append_time(h, t))
            h = self.act(h + h_new)
        return self.layers[-1](self._append_time(h, t))
    
    
class ResNet2(nn.Module):
    def __init__(self, input_dim, output_dim, dt, t0,
                 hidden_dim=512,
                 time_embed_dim=128,
                 num_res_blocks=6, dtype=torch.float32):
        super(ResNet2, self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.dt, self.t0 = dt, t0
        self.dtype = dtype
        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = FCnet(data_dim=input_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
        self.norm = nn.GroupNorm(32, hid, dtype=self.dtype)

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid, hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid, output_dim, dtype=self.dtype),)

    def forward(self, x, v, t):
        # make sure t.shape = [T]
        if len(t.shape) == 0:
            t = t[None]
        t = (t-self.t0) / self.dt
        u = torch.cat([x, v], dim=-1)
        t_emb = timestep_embedding(t, self.time_embed_dim).to(self.dtype)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(u)
        out = self.out_module(self.norm(x_out + t_out))

        return out    


# class ResNet2(nn.Module):
#     def __init__(self, g_dim, xi_dim, dt, t0,
#                  hidden_dim=256,
#                  time_embed_dim=128,
#                  num_res_blocks=6, dtype=torch.float32):
#         super(ResNet2, self).__init__()

#         self.time_embed_dim = time_embed_dim
#         hid = hidden_dim
#         self.dt, self.t0 = dt, t0
#         self.dtype = dtype
#         self.t_module = nn.Sequential(
#             nn.Linear(self.time_embed_dim, hid),
#             SiLU(),
#             nn.Linear(hid, hid),
#         )
#         self.g_module = FCnet(data_dim=g_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
#         self.xi_module = FCnet(data_dim=xi_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
#         self.norm = nn.GroupNorm(32, hid, dtype=self.dtype)

#         self.out_module = nn.Sequential(
#             nn.Linear(hid, hid, dtype=self.dtype),
#             SiLU(),
#             nn.Linear(hid, xi_dim, dtype=self.dtype),)

#     def forward(self, g, xi, t):
#         # make sure t.shape = [T]
#         if len(t.shape) == 0:
#             t = t[None]
#         t = (t-self.t0) / self.dt
#         t_emb = timestep_embedding(t, self.time_embed_dim).to(self.dtype)
#         t_out = self.t_module(t_emb)
#         g_out = self.g_module(g)
#         xi_out = self.xi_module(xi)
#         out = self.out_module(self.norm(g_out + xi_out + t_out))

#         return out

class FCnet(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.map = nn.Linear(data_dim, hidden_dim, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hidden_dim, dtype=self.dtype)

        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

        self.norms = nn.ModuleList(
            # [nn.BatchNorm1d(hidden_dim) for _ in range(num_res_blocks)]
            [nn.GroupNorm(32, hidden_dim, dtype=self.dtype) for _ in range(num_res_blocks)])

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths = [hid]*4
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i+1], dtype=self.dtype))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.map(x)
        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / np.sqrt(2)
        return h
