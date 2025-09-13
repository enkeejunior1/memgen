# Simple multi-model training for cloneofsimo and other simple minded people
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import RF, SimpleMLP, GaussianMixtureDataset, sample_batch, seed_everything
from functools import partial

def main():
    # Simple configs - just what we need
    base_width = 256
    n_samples = [32768] # 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    widths = [256, 512, 1024, 2048, 4096, 8192]
    lrs = [2**-i for i in range(10)]
    
    k = 1
    batch_size = 256
    iterations = 50000
    eval_interval = 1000

    val_t_list = [round(x, 1) for x in np.linspace(0.1, 1, 10).tolist()]
    print(f"Training {len(widths)} models x {len(lrs)} LRs x {len(n_samples)} samples = {len(widths) * len(lrs) * len(n_samples)} experiments")
    
    # Initialize wandb
    # import wandb
    # wandb.login(key="348c7069db9b04fc286e83138f5052d0492b52c5")
    # wandb.init(project="rf-mup-sweep", name="data_size_sweep")
    os.makedirs("contents", exist_ok=True)
    
    # Create dataset once
    ds = GaussianMixtureDataset(k=k, n_samples=n_samples)
    
    # Store all models and their info
    rfs = {}
    models = {}
    optimizers = {}
    results = {}
    samplers = {}
    val_sampler = lambda: sample_batch(ds.data, ds.labels, batch_size)
    
    # Create all models
    for n_sample in n_samples:
        for width in widths:
            for lr in lrs:
                name = f"n{n_sample}_w{width}_lr{lr:.0e}"
                
                # Create model with μP
                model = SimpleMLP(in_dim=2, out_dim=2, width=width, depth=5).cuda()
                rf = RF(model)
                
                # μP learning rate scaling
                input_lr = lr / np.sqrt(width)
                hidden_lr = lr / np.sqrt(width)
                output_lr = lr
                
                # μP optimizer setup - output layer gets different learning rate
                param_groups = [
                    {'params': model.layers[0].parameters(), 'lr': input_lr},  # μP scaled lr for input
                    {'params': model.layers[1:-1].parameters(), 'lr': hidden_lr},  # μP scaled lr for hidden
                    {'params': model.layers[-1].parameters(), 'lr': output_lr},  # μP scaled lr for output
                ]
                optimizer = optim.Adam(param_groups)
                
                rfs[name] = rf
                models[name] = model
                optimizers[name] = optimizer
                results[name] = []

                data = ds.data[:n_sample] if n_sample > batch_size else ds.data.repeat(batch_size // n_sample, 1)
                labels = ds.labels[:n_sample] if n_sample > batch_size else ds.labels.repeat(batch_size // n_sample)
                samplers[name] = partial(
                    sample_batch, data=data, labels=labels, batch_size=batch_size, 
                )
                print(f"Created {name}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Total GPU memory: ~{sum(sum(p.numel() for p in m.parameters()) for m in models.values()) * 4 / 1e9:.2f} GB")
    
    # Training loop
    for iteration in tqdm(range(iterations), desc="Training"):
        # Train each model on the same batch
        for name in models:
            x, c = samplers[name]()
            x, c = x.cuda(), c.cuda()
            c = torch.nn.functional.one_hot(c, num_classes=k*k).float()
            
            optimizers[name].zero_grad()
            loss, _ = rfs[name].forward(x, c)
            loss.backward()
            optimizers[name].step()
        
        # Validation every eval_interval iterations
        if iteration % eval_interval == 0:
            val_losses = {}
            val_losses_per_t = {t_idx: {} for t_idx, t in enumerate(val_t_list)}
            with torch.no_grad():
                for name in models:
                    x, c = val_sampler(generator=torch.Generator().manual_seed(42 + iteration))
                    x, c = x.cuda(), c.cuda()
                    c = torch.nn.functional.one_hot(c, num_classes=k*k).float()
                    
                    # for t_idx, t in enumerate(val_t_list):
                    #     seed_everything(42)
                    #     val_loss, _ = rfs[name].forward(x, c, t=t)
                    #     val_losses_per_t[t_idx][name] = val_loss.item()
                    
                    seed_everything(42)
                    val_losses[name] = rfs[name].forward(x, c)[0].item()

            # Save val_losses_per_t, val_losses
            os.makedirs(f"contents/{iteration}", exist_ok=True)
            # with open(f"contents/{iteration}/val_losses_per_t.pkl", "wb") as f:
            #     pickle.dump(val_losses_per_t, f)
            # with open(f"contents/{iteration}/val_losses.pkl", "wb") as f:
            #     pickle.dump(val_losses, f)
            
            # Create plots
            # create_plots_plot_1(n_samples, widths, lrs, val_losses, file_name=f"contents/{iteration}/total_t")
            create_plots_plot_2(n_samples, widths, lrs, val_losses, file_name=f"contents/{iteration}/total_t")

            # for t_idx, t in enumerate(val_t_list):
            #     create_plots_plot_1(n_samples, widths, lrs, val_losses_per_t[t_idx], file_name=f"contents/{iteration}/t{t}")
            #     create_plots_plot_2(n_samples, widths, lrs, val_losses_per_t[t_idx], file_name=f"contents/{iteration}/t{t}")

        if iteration == 0:
            print(f"RUN GPU memory: ~{sum(sum(p.numel() for p in m.parameters()) for m in models.values()) * 4 / 1e9:.2f} GB")
    
def create_plots_plot_1(n_samples, widths, lrs, val_losses, file_name=""):
    """Create the two requested plots"""
    # Plot 1: Data size vs Validation Loss (for each width and LR combination)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each width-lr combination, plot data size vs validation loss
    colors = plt.cm.tab20(np.linspace(0, 1, len(widths) * len(lrs)))
    color_idx = 0
    
    for width in widths:
        for lr in lrs:
            data_sizes_i = []
            val_losses_i = []
            
            for n_sample in n_samples:
                name = f"n{n_sample}_w{width}_lr{lr:.0e}"
                if name in val_losses:
                    data_sizes_i.append(n_sample)
                    val_losses_i.append(val_losses[name])
            
            if data_sizes_i and val_losses_i:
                ax.plot(
                    data_sizes_i, val_losses_i,
                    marker='o', linewidth=1.5, markersize=4, 
                    color=colors[color_idx],
                    label=f'W{width}, LR{lr:.0e}', alpha=0.7
                )
                color_idx += 1
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Data Size', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Data Size vs Validation Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{file_name}-data_size_vs_val_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_plots_plot_2(n_samples, widths, lrs, val_losses, file_name=""):
    # Plot 2: Learning Rate vs Validation Loss (for each width and data size combination)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each width-data_size combination, plot LR vs validation loss
    colors = plt.cm.tab20(np.linspace(0, 1, len(widths) * len(n_samples)))
    color_idx = 0
    
    for width in widths:
        for n_sample in n_samples:
            learning_rates_i = []
            val_losses_i = []
            
            for lr in lrs:
                name = f"n{n_sample}_w{width}_lr{lr:.0e}"
                if name in val_losses:
                    learning_rates_i.append(lr)
                    val_losses_i.append(val_losses[name])
            
            if learning_rates_i and val_losses_i:
                # Sort by learning rate
                sorted_data = sorted(zip(learning_rates_i, val_losses_i))
                learning_rates_i, val_losses_i = zip(*sorted_data)
                
                ax.plot(
                    learning_rates_i, val_losses_i,
                    marker='o', linewidth=1.5, markersize=4,
                    color=colors[color_idx],
                    label=f'W{width}, N{n_sample}', alpha=0.7
                )
                color_idx += 1
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Learning Rate vs Validation Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{file_name}-lr_vs_val_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
