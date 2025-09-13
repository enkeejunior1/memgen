# Simple multi-model training for cloneofsimo and other simple minded people
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from train import SimpleMLP, RF, GaussianMixtureDataset

def main():
    # Simple configs - just what we need
    base_width = 256
    widths = [256, 512, 1024, 2048, 4096]
    lrs = [1e-6, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    n_samples = 50000
    epochs = 50
    k = 5
    print(f"Training {len(widths)} models x {len(lrs)} LRs = {len(widths) * len(lrs)} experiments")
    
    # Initialize wandb
    wandb.login(key="348c7069db9b04fc286e83138f5052d0492b52c5")
    wandb.init(project="rf-mup-sweep", name="simple_multi_sweep")
    
    # Create dataset once
    ds = GaussianMixtureDataset(k=k, n_samples=n_samples)
    dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
    
    # Store all models and their info
    models = {}
    optimizers = {}
    rfs = {}
    results = {}
    
    # Create all models
    for width in widths:
        for lr in lrs:
            name = f"w{width}_lr{lr:.0e}"
            
            # Create model with μP
            model = SimpleMLP(width=width, depth=5, mup=True, base_width=base_width).cuda()
            rf = RF(model)
            
            # μP learning rate scaling
            width_mult = width / base_width
            mup_lr = lr / width_mult
            
            # Simple optimizer setup
            param_groups = [
                {'params': model.output_layer.parameters(), 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if 'output_layer' not in n], 'lr': mup_lr}
            ]
            optimizer = optim.Adam(param_groups)
            
            models[name] = model
            rfs[name] = rf
            optimizers[name] = optimizer
            results[name] = []
            print(f"Created {name}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Total GPU memory: ~{sum(sum(p.numel() for p in m.parameters()) for m in models.values()) * 4 / 1e9:.2f} GB")
    
    # Training loop
    for epoch in range(epochs):
        # Train all models
        epoch_losses = {}
        
        for x, c in tqdm(dl, desc=f"Epoch {epoch}"):
            x, c = x.cuda(), c.cuda()
            
            # Train each model on the same batch
            for name in models:
                optimizers[name].zero_grad()
                loss, _ = rfs[name].forward(x, c)
                loss.backward()
                optimizers[name].step()
                
                if name not in epoch_losses:
                    epoch_losses[name] = []
                epoch_losses[name].append(loss.item())
        
        # Calculate averages and store
        for name in models:
            avg_loss = np.mean(epoch_losses[name])
            results[name].append(avg_loss)
        
        # Print top 5 performers
        current_losses = {name: results[name][-1] for name in models}
        top_5 = sorted(current_losses.items(), key=lambda x: x[1])[:5]
        
        print(f"\n📈 Epoch {epoch} - Top 5:")
        for i, (name, loss) in enumerate(top_5, 1):
            print(f"  {i}. {name}: {loss:.6f}")
        
        # Log to wandb
        log_dict = {"epoch": epoch}
        for name, loss in current_losses.items():
            log_dict[f"loss_{name}"] = loss
        wandb.log(log_dict)
    
    # Final results - Focus on validation loss vs LR
    final_losses = {name: results[name][-1] for name in models}
    
    # Organize results by width and LR for plotting
    width_lr_results = {}
    for width in widths:
        width_lr_results[width] = []
        for lr in lrs:
            name = f"w{width}_lr{lr:.0e}"
            if name in final_losses:
                width_lr_results[width].append((lr, final_losses[name]))
        # Sort by LR
        width_lr_results[width].sort(key=lambda x: x[0])
    
    # Create the μP-style plot: LR vs Validation Loss
    os.makedirs("contents", exist_ok=True)
    
    # Plot each width as a separate line
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(widths)))
    for i, width in enumerate(widths):
        if width in width_lr_results and width_lr_results[width]:
            lrs_for_width = [x[0] for x in width_lr_results[width]]
            losses_for_width = [x[1] for x in width_lr_results[width]]
            
            ax.plot(
                lrs_for_width, losses_for_width, 
                marker='o', linewidth=2, markersize=6, color=colors[i], 
                label=f'Width {width}'
            )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('μP Style: Learning Rate vs Validation Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("contents/mup_style_lr_vs_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
