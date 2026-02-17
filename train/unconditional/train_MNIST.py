import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import Iterable, List
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import torch.nn.functional as F

from models.JiT import JiTUncond_XS_4
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"
title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def compute_V(x, y_pos, y_neg, tau):
    # x: [N, D]
    # y_pos: [N_sample, D]
    # y_neg: [N_sample, D]
    # tau: temperature
    N = x.shape[0]
    N_pos = y_pos.shape[0]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    # ignore self (if y_neg is x) 
    if x.shape == y_neg.shape:
        dist_neg += torch.eye(N, device=x.device) * 1e6

    # compute logits
    logit_pos = -dist_pos / tau
    logit_neg = -dist_neg / tau

    # concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)

    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)

    A = torch.sqrt(A_row * A_col)

    # back to [N, N_sample]
    A_pos, A_neg = torch.split(A, [N_pos, A.shape[1] - N_pos], dim=1)

    # compute weights
    W_pos = A_pos
    W_neg = A_neg
    W_pos = W_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = W_neg * A_pos.sum(dim=1, keepdim=True)

    drift_pos = W_pos @ y_pos 
    drift_neg = W_neg @ y_neg

    V = drift_pos - drift_neg
    return V


def drifting_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    taus: Iterable[float] = (0.02, 0.05, 0.2, 1, 2, 5, 10, 20),
) -> torch.Tensor:
    # Flatten [B, C, H, W] -> [N, D]
    x = gen.view(gen.shape[0], -1)
    y = pos.view(pos.shape[0], -1)
    
    total_loss = 0.0
    for tau in taus:
        V = compute_V(x, y, x, tau)
        # Author Eq. 26: loss = MSE(x, sg(x + V))
        target = (x + V).detach()
        total_loss += F.mse_loss(x, target)
        
    return total_loss


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a drift model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_pref', type=str, default='unconditional', help='Prefix for saved directory')
    parser.add_argument('--save_dir', type=str, default='drift-MNIST', help='Directory to save models')
    # parser.add_argument('--mae_ckpt', type=str, default='saved_runs/mae/mae-MNIST/checkpoints/best_mae.tar', help='Path to a MAE checkpoint to load')

    args = parser.parse_args()
    batch_size = 1024
    min_lr = 1e-4
    max_lr = 5e-4

    epochs = 600
    max_steps = 400000
    num_workers = 16
    save_dir = args.save_dir
    save_pref = args.save_pref
    # mae_ckpt = args.mae_ckpt

    device = args.device

    DATA_ROOT = Path("data").resolve()


    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    # Load MNIST dataset
    train_dataset = MNIST(
        root=str(DATA_ROOT), 
        transform=transform,
        # split="train",
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # mae = MAE(in_ch_pixel=3, unet_ch=128, unet_groups=32).to(device)
    # mae = mae.to(device)

    # mae_checkpoint = torch.load(
    #     mae_ckpt,
    #     map_location=device,
    # )

    # mae_ema_model = AveragedModel(mae, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    # mae_ema_model.load_state_dict(mae_checkpoint["ema_state_dict"])
    # mae.load_state_dict(mae_ema_model.module.state_dict())
    # for p in mae.parameters():
    #     p.requires_grad = False
    # mae.eval()

    model = JiTUncond_XS_4(in_channels=1, input_size=32).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    optim = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    # after loading the data we change working directory

    os.makedirs(f"saved_runs/{save_pref}/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_pref}/{save_dir}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    ckpt = args.ckpt
    ema_state_dict = None
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) 
        ema_state_dict = checkpoint.get("ema_state_dict")
    else:
        curr_epoch = 0

    model = torch.compile(model)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_model = AveragedModel(
        base_model,
        multi_avg_fn=get_ema_multi_avg_fn(0.999),
    ).to(device)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    # Initialize best loss tracking
    best_loss = float('inf')
    loss_history = []
    
    for epoch in pbar:
        model.train()
        epoch_losses = []
        
        # Use a fixed noise set for sampling to monitor progress consistently, 
        # or different noise to check diversity. 
        # Let's ensure the noise is generated fresh each epoch.
        
        for i, (x, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False, total=len(train_loader)):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            
            # # I think bf16 is reasonable here with the normalization, but I want to see if Kaiming's group uses it here
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                eps = torch.randn((x.shape[0] // 2, x.shape[1], x.shape[2], x.shape[3]), device=device)
                pred = model(eps)
                # loss = drifting_loss(pred, x, mae=mae)
                loss = drifting_loss(pred, x)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            
            grad = torch.norm(
                torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
            )

            optim.step()
            ema_model.update_parameters(model._orig_mod if hasattr(model, "_orig_mod") else model)
            scheduler.step()

            true_loss = loss.item()
            epoch_losses.append(true_loss)
        
        # Calculate epoch average loss
        avg_epoch_loss = np.mean(epoch_losses)
        loss_history.append(avg_epoch_loss)
        pbar.set_postfix_str(f'Epoch: {epoch} | Loss: {avg_epoch_loss:.10e} | Grad: {grad.item():.10e}')
      
        model.eval()
        ema_model.eval()
        with torch.no_grad():
            C, H, W = x.shape[1:]
            
            # Use different noise for each sample in the batch to check for collapse
            sample_noise = torch.randn((4, C, H, W), device=device)
            pred = ema_model(sample_noise)
            
            # Create combined figure with samples and loss plot
            fig = plt.figure(figsize=(16, 10))
            
            # Create subplot grid: 2 rows, 4 columns with more spacing
            gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.4, 
                                 left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # Plot samples (top row)
            sample_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
            for i in range(4):
                # MNIST is grayscale: [1, 32, 32] -> [32, 32]
                img = pred[i].squeeze().cpu().clamp(0, 1).numpy()
                sample_axes[i].imshow(img, cmap='gray')
                sample_axes[i].axis("off")
            
            # Add loss plot on bottom row (spanning all 4 columns)
            ax_loss = fig.add_subplot(gs[1, :])
            ax_loss.semilogy(loss_history, 'b-', linewidth=2)
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss (log scale)')
            ax_loss.set_title(f'Training Loss - Epoch {epoch}')
            ax_loss.grid(True, alpha=0.3)
            
            plt.savefig(f'samples/epoch_{epoch}.jpg', dpi=75, bbox_inches='tight')
            plt.close() 

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint = {
                "epoch": int(epoch),
                "model_state_dict": model._orig_mod.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "best_loss": best_loss
            }
            torch.save(checkpoint, f'checkpoints/best_model.tar')
    
    # Save final model
    final_checkpoint = {
        "epoch": int(epochs),
        "model_state_dict": model._orig_mod.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "final_loss": avg_epoch_loss
    }
    torch.save(final_checkpoint, f'checkpoints/final_model.tar')
    print(f"Saved final model with loss: {avg_epoch_loss:.6f}")
     