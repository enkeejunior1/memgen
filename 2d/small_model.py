# implementation of Rectified Flow for simple minded people like me.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 2D dataset 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class GaussianMixtureDataset(Dataset):
    def __init__(self, k=3, n_samples=10000):
        self.k = k
        self.n_samples = n_samples
        
        # Create k x k grid of gaussian centers
        centers = []
        for i in range(k):
            for j in range(k):
                # Map centers to [-1, 1] range
                x = -1 + 2 * i / (k - 1) if k > 1 else 0
                y = -1 + 2 * j / (k - 1) if k > 1 else 0
                centers.append([x, y])
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.n_samples):
            # Randomly select a center
            center_idx = torch.randint(0, len(self.centers), (1,))
            center = self.centers[center_idx]
            
            # Sample from gaussian around that center
            sample = center + 0.3 * torch.randn(2)  # std=0.3
            data.append(sample)
        return torch.stack(data)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(0)  # dummy label

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
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
    def __init__(self, width=128, depth=5):
        super().__init__()
        self.width = width
        
        # Input: 2D point only
        self.input_layer = nn.Linear(2, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(width, 2)  # Output: 2D velocity
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(width)
        
    def forward(self, x, t, cond=None):
        # Get timestep embedding
        t_emb = self.t_embedder(t)
        
        # Forward pass with residual connections
        h = F.gelu(self.input_layer(x))
        h = h + t_emb
        for hidden in self.hidden:
            h = h + F.gelu(hidden(h))
        out = self.output_layer(h)
        return out

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
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
        samples = [z.clone()]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            samples.append(z.clone())
        return samples


if __name__ == "__main__":
    # train RF on 2D gaussian mixture
    import torch.optim as optim
    from tqdm import tqdm
    import os

    # Create model
    model = SimpleMLP(width=128, depth=4).cuda()
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create dataset
    dataset = GaussianMixtureDataset(k=3, n_samples=50000)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    os.makedirs("contents", exist_ok=True)

    for epoch in range(100):
        total_loss = 0
        count = 0
        
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch: {epoch}, Loss: {avg_loss:.6f}")

        # Sample and visualize every 10 epochs
        if epoch % 10 == 0:
            rf.model.eval()
            with torch.no_grad():
                # Sample from noise
                init_noise = torch.randn(1000, 2).cuda()
                cond = torch.zeros(1000).cuda()  # dummy condition
                
                samples = rf.sample(init_noise, cond, sample_steps=50)
                
                # Plot original data and samples
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Original data
                original_data = dataset.data.numpy()
                ax1.scatter(original_data[:1000, 0], original_data[:1000, 1], alpha=0.5, s=1)
                ax1.set_title("Original Data")
                ax1.set_aspect('equal')
                
                # Generated samples
                final_samples = samples[-1].cpu().numpy()
                ax2.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, s=1)
                ax2.set_title(f"Generated Samples (Epoch {epoch})")
                ax2.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(f"contents/samples_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
                plt.close()

            rf.model.train()
