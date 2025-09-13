import torch
import torch.nn as nn
import math
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond, t=None):
        b = x.size(0)
        if t is not None:
            if isinstance(t, float):
                t = torch.tensor([t] * b).to(x.device)
            else:
                raise ValueError(f"t must be a float, got {type(t)}")
        else:
            if self.ln:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
            else:
                t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn(x.shape).to(x.device)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

class ScaledLinear(nn.Module):
    def __init__(self, in_dim, out_dim, weight_type):
        super().__init__()
        assert weight_type in ['in', 'hidden', 'out']
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.fan_in = in_dim
        self.fan_out = out_dim

        # forward / backward scaling factor
        if weight_type == 'in':
            self.scale_forward = 1.0
        elif weight_type == 'hidden':
            self.scale_forward = 1.0 / math.sqrt(self.fan_in)
        elif weight_type == 'out':
            assert in_dim % 2 == 0
            assert self.weight.data.shape == (self.fan_out, self.fan_in)
            with torch.no_grad():
                self.weight.data[:, -in_dim//2:] = -self.weight.data[:, :in_dim//2]
            self.scale_forward = 1.0 / self.fan_in
            self.scale_backward = 1.0 / math.sqrt(self.fan_out)
            def hook_fn(grad):
                return grad * self.scale_backward
            self.weight.register_hook(hook_fn)

    def forward(self, x):
        # forward: apply 1/sqrt(fan_in) scaling
        return torch.nn.functional.linear(x, self.weight * self.scale_forward, self.bias)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ScaledLinear(frequency_embedding_size, hidden_size, weight_type='in'),
            nn.SiLU(),
            ScaledLinear(hidden_size, hidden_size, weight_type='hidden'),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=256, depth=5, num_classes=1):
        super().__init__()
        layers = []
        dims = [in_dim] + [width] * (depth - 2) + [out_dim]
        layers.append(ScaledLinear(
            dims[0], dims[1], weight_type='in'
        ))
        for i in range(1, len(dims) - 2):
            layers.append(ScaledLinear(
                dims[i], dims[i+1], weight_type='hidden'
            ))
        layers.append(ScaledLinear(
            dims[-2], dims[-1], weight_type='out'
        ))
        self.layers = nn.ModuleList(layers)
        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = ScaledLinear(num_classes, width, weight_type='in')
        self.nonlinearity = nn.SiLU()

    def forward(self, x, t, y):
        x = self.layers[0](x) + self.t_embedder(t) + self.y_embedder(y)
        for layer in self.layers[1:-1]:
            x = x + 1 / np.sqrt(len(self.layers[1:-1])) * self.nonlinearity(layer(x))
        return self.layers[-1](x)


def sample_batch(data, labels, batch_size, generator=None):
    random_indices = torch.randperm(len(data), generator=generator)[:batch_size]
    return data[random_indices], labels[random_indices]


class GaussianMixtureDataset(Dataset):
    def __init__(self, k=5, n_samples=50000):
        self.k = k
        if isinstance(n_samples, list):
            self.n_samples = max(n_samples)
        else:
            self.n_samples = n_samples
        
        # Create k x k grid of centers uniformly distributed in [-1, 1]
        grid_size = int(np.sqrt(k))
        if grid_size * grid_size != k:
            raise ValueError(f"k={k} must be a perfect square for k x k grid")
        
        if grid_size == 1:
            # Special case for k=1: center at (0, 0)
            xx = torch.tensor([[0.0]])
            yy = torch.tensor([[0.0]])
        else:
            x = torch.linspace(-1, 1, grid_size)
            y = torch.linspace(-1, 1, grid_size)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Generate data
        self.data = []
        self.labels = []
        
        for i in range(self.n_samples):
            class_idx = torch.randint(0, k, (1,)).item()
            sample = torch.randn(2) * 0.1 + self.centers[class_idx]
            self.data.append(sample)
            self.labels.append(class_idx)
        
        self.data = torch.stack(self.data).float()
        self.labels = torch.tensor(self.labels).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]