import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import Iterable, List
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.datasets import CelebA

import torch.nn.functional as F

from models.JiT import JiTUncond_B_16
from models.MAE import MAE
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

# -------------------------
# Algorithm-2 drift, but reusing precomputed distances (paper: "reuse cdist")
# -------------------------
def compute_drift_from_dists(
    x: torch.Tensor,                 # [N, C]
    y_pos: torch.Tensor,             # [M, C]
    dist_pos: torch.Tensor,          # [N, M] = cdist(x, y_pos)
    dist_neg: torch.Tensor,          # [N, N] = cdist(x, x) with diagonal masked
    tau_tilde: float,                # \tilde{\tau}_j = tau * sqrt(C_j)
) -> torch.Tensor:
    """
    Implements Alg.2 exactly, except it takes distances as input so we can
    reuse cdist for S_j + all temperatures.
    """
    N = x.shape[0]
    M = y_pos.shape[0]
    M_plus_N = dist_pos.shape[1] + dist_neg.shape[1]

    logit = torch.cat([-dist_pos / tau_tilde, -dist_neg / tau_tilde], dim=1)  # [N, M+N]
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = torch.sqrt(A_row * A_col)                                            # [N, M+N]

    A_pos, A_neg = torch.split(A, [dist_pos.shape[1], dist_neg.shape[1]], dim=1)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)                           # [N, M]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)                           # [N, N]

    drift_pos = W_pos @ y_pos                                                # [N, C]
    drift_neg = W_neg @ x                                                    # [N, C]   (y_neg = x)
    return drift_pos - drift_neg                                              # [N, C]


# -------------------------
# S_j and lambda_j from the paper (Eqs. 21 and 25)
# computed via batch empirical mean, with stopgrad.
# -------------------------
@torch.no_grad()
def estimate_S_from_dists(dist_pos_raw: torch.Tensor, sqrt_C: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Eq. (21): S_j = (1/sqrt(C_j)) * E || phi(x) - phi(y) ||
    Here we use the batch empirical mean in place of expectation,
    using the same cdist matrix dist_pos_raw = cdist(phi(x), phi(y_pos)).
    """
    mean_dist = dist_pos_raw.mean()
    return (mean_dist / (sqrt_C + eps)).clamp_min(eps)


@torch.no_grad()
def estimate_lambda(V: torch.Tensor, C: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Eq. (25): lambda_j = sqrt( E[ (1/C_j) ||V_j||^2 ] )
    Again use batch empirical mean.
    """
    mean_sq = (V.norm(dim=1) ** 2).mean()
    return torch.sqrt(mean_sq / float(C) + eps).clamp_min(eps)


# -------------------------
# Helper: turn MAEFeatures items into [N, C] samples
# - LocationWiseFeatures: flatten all spatial locations across batch
# - PooledFeatures: flatten pooled vectors across batch
# This matches "Normalization across spatial locations" in the screenshot.
# -------------------------
def flatten_feature_item(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        # [B, C, H, W] -> [B*H*W, C]
        B, C, H, W = t.shape
        return t.permute(0, 2, 3, 1).reshape(-1, C)
    if t.dim() == 3:
        # [B, Npool, C] -> [B*Npool, C]
        B, Np, C = t.shape
        return t.reshape(-1, C)

    raise ValueError(f"Expected dim 3 or 4, got {t.shape}")



# -------------------------
# Paper-faithful drifting loss (Eqs. 18-26 + Multiple temperatures)
# - For each feature j:
#   1) compute S_j from cdist(phi(x), phi(y))  (Eq.21)
#   2) define normalized feature phi~ = phi / S_j (Eq.18)
#   3) for each tau:
#        tau_tilde = tau * sqrt(C_j)                 (Eq.22)
#        compute V_{j,tau} using Alg.2 on normalized distances
#        normalize drift: V~_{j,tau} = V_{j,tau} / lambda_{j,tau}  (Eqs.23-25)
#      aggregate drift: V~_j = sum_tau V~_{j,tau}      ("Multiple temperatures")
#   4) loss L_j = MSE(phi~(x), sg(phi~(x) + V~_j))     (Eq.26)
# - total loss = sum_j L_j
# -------------------------
def drifting_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    mae: torch.nn.Module,
    taus: Iterable[float] = (0.02, 0.05, 0.2),
    eps_diag: float = 1e6,
) -> torch.Tensor:
    mae.eval()

    # pos features: no grad
    with torch.no_grad():
        f_pos = mae._get_features(pos)

    # gen features: needs grad
    f_gen = mae._get_features(gen)

    def loss_over_feature_lists(gen_list: List[torch.Tensor], pos_list: List[torch.Tensor]) -> torch.Tensor:
        total = gen.new_tensor(0.0)

        for g_item, p_item in zip(gen_list, pos_list):
            x = flatten_feature_item(g_item)  # [N, Cj]
            y = flatten_feature_item(p_item)  # [M, Cj]
            Cj = x.shape[1]
            sqrt_C = float(Cj) ** 0.5

            with torch.no_grad():
                x_ng = x.detach().half()
                y_ng = y.detach().half()
                dist_pos_raw = torch.cdist(x_ng, y_ng)                                 # [N, M] (half)
                dist_neg_raw = torch.cdist(x_ng, x_ng)                                 # [N, N] (half)
                
                # In-place diagonal mask
                dist_neg_raw.diagonal().add_(eps_diag)
                
                # paper: y denotes positive/negative sample -> use both dists
                dist_all_raw = torch.cat([dist_pos_raw, dist_neg_raw], dim=1)           # [N, M+N]
                # Eq.(21): S_j from empirical mean of dist_all_raw
                S_j = estimate_S_from_dists(dist_all_raw, sqrt_C=sqrt_C)

            # Eq.(18): normalized features
            x_tilde = x / S_j
            y_tilde = y / S_j

            # normalized distances (Eq.19): dist_tilde = dist_raw / S_j
            # we reuse dist_pos_raw/dist_neg_raw in-place
            dist_pos_raw.div_(S_j)
            dist_neg_raw.div_(S_j)
            dist_pos_tilde = dist_pos_raw
            dist_neg_tilde = dist_neg_raw

            # Multiple temperatures: sum *normalized* drifts
            V_tilde_sum = x_tilde.new_zeros(x_tilde.shape)

            for tau in taus:
                # Eq.(22): tau_tilde = tau * sqrt(Cj)
                tau_tilde = float(tau) * sqrt_C

                # Alg.2 drift in normalized feature space
                V = compute_drift_from_dists(
                    x=x_tilde,
                    y_pos=y_tilde,
                    dist_pos=dist_pos_tilde,
                    dist_neg=dist_neg_tilde,
                    tau_tilde=tau_tilde,
                )  # [N, Cj]

                # Eqs.(23-25): lambda and normalized drift
                with torch.no_grad():
                    lam = estimate_lambda(V.detach(), C=Cj)
                V_tilde = V / lam

                V_tilde_sum = V_tilde_sum + V_tilde

            # Eq.(26): one MSE per feature j, using aggregated drift
            with torch.no_grad():
                target = (x_tilde + V_tilde_sum).detach()
            total = total + F.mse_loss(x_tilde, target, reduction="mean")

        return total

    loss = gen.new_tensor(0.0)
    loss += loss_over_feature_lists(f_gen.LocationWiseFeatures, f_pos.LocationWiseFeatures)
    loss += loss_over_feature_lists(f_gen.PooledFeatures, f_pos.PooledFeatures)
    loss += loss_over_feature_lists(gen, pos)
    return loss


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a drift model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_pref', type=str, default='unconditional', help='Prefix for saved directory')
    parser.add_argument('--save_dir', type=str, default='drift-celeba', help='Directory to save models')
    parser.add_argument('--mae_ckpt', type=str, default='saved_runs/mae/mae-celeba/checkpoints/best_mae.tar', help='Path to a MAE checkpoint to load')

    args = parser.parse_args()
    batch_size = 32
    min_lr = 1e-4
    max_lr = 5e-4

    epochs = 600
    max_steps = 400000
    num_workers = 16
    save_dir = args.save_dir
    save_pref = args.save_pref
    mae_ckpt = args.mae_ckpt

    device = args.device

    DATA_ROOT = Path("data").resolve()


    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    # Load CelebA dataset
    train_dataset = CelebA(
        root=str(DATA_ROOT), 
        transform=transform,
        split="train",
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

    mae = MAE(in_ch_pixel=3, unet_ch=128, unet_groups=32).to(device)
    mae = mae.to(device)

    mae_checkpoint = torch.load(
        mae_ckpt,
        map_location=device,
    )

    mae_ema_model = AveragedModel(mae, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    mae_ema_model.load_state_dict(mae_checkpoint["ema_state_dict"])
    mae.load_state_dict(mae_ema_model.module.state_dict())
    for p in mae.parameters():
        p.requires_grad = False
    mae.eval()

    model = JiTUncond_B_16(in_channels=3, input_size=128).to(device)

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
        
        for i, (x, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            
            # # I think bf16 is reasonable here with the normalization, but I want to see if Kaiming's group uses it here
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                eps = torch.randn((x.shape[0] // 2, x.shape[1], x.shape[2], x.shape[3]), device=device)
                pred = model(eps)
                loss = drifting_loss(pred, x, mae=mae)

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
            
            eps = torch.randn((4, C, H, W), device=device)
            pred = ema_model(eps)
            
            # Create combined figure with samples and loss plot
            fig = plt.figure(figsize=(16, 10))
            
            # Create subplot grid: 2 rows, 4 columns with more spacing
            gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.4, 
                                 left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # Plot samples (top row)
            sample_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
            for i in range(4):
                # Handle 3-channel RGB: [C, H, W] -> [H, W, C]
                img = pred[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                sample_axes[i].imshow(img)
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
     