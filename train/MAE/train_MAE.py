import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import argparse
import matplotlib.pyplot as plt
from models.MAE import MAE

plt.rcParams["font.family"] = "DejaVu Serif"
title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def mae_masked_mse(x_recon, x, mask):
    # mask: bool [B,1,H,W] or [B,C,H,W]
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1, -1)

    mask_f = mask.to(dtype=x.dtype)
    diff2 = (x_recon - x).pow(2)

    denom = mask_f.sum().clamp_min(1.0)
    return (diff2 * mask_f).sum() / denom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MAE feature encoder.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_pref", type=str, default="mae")
    parser.add_argument("--save_dir", type=str, default="mae-circles")
    parser.add_argument("--data_file", type=str, default="eit-circles-dtn-default-128.pt")
    args = parser.parse_args()

    batch_size = 256
    lr = 4e-3

    epochs = 3000
    num_workers = 16

    mask_ratio = 0.5   
    ema_decay = 0.9995

    device = args.device

    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")
    dataset_train = dataset["train"]["media"]
    if dataset_train.ndim == 3:
        dataset_train = dataset_train.unsqueeze(1)  # [N,1,H,W]

    dataset_train = dataset_train.contiguous(memory_format=torch.channels_last)

    train = TensorDataset(dataset_train.detach().clone())
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=16,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    mae = MAE(in_ch_pixel=1, mask_ratio=mask_ratio, spatial_downsample=[8, 2, 2]).to(device)
    mae = mae.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params/1e6:.6f}M")

    optim = torch.optim.AdamW(
        mae.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=5e-2,
    )

    os.makedirs(f"saved_runs/{args.save_pref}/{args.save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{args.save_pref}/{args.save_dir}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    curr_epoch = 0
    ema_state_dict = None
    best_loss = float("inf")

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        mae.load_state_dict(checkpoint["model_state_dict"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        ema_state_dict = checkpoint.get("ema_state_dict")
        best_loss = checkpoint.get("best_loss", best_loss)

    mae = torch.compile(mae)

    base_model = mae._orig_mod if hasattr(mae, "_orig_mod") else mae
    ema_model = AveragedModel(
        base_model,
        multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
    ).to(device)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="MAE Epochs")
    loss_history = []
    vis_every = 100
    x_vis_cache = None

    for epoch in pbar:
        mae.train()
        epoch_losses = []

        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )

            if (epoch % vis_every == 0) and i == 0:
                x_vis_cache = x[:4].detach()

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = mae(x) 
                loss = mae_masked_mse(out.x_recon, x, out.mask)

            if not torch.isfinite(loss):
                continue

            loss.backward()

            grad = torch.norm(
                torch.stack([p.grad.norm(2) for p in mae.parameters() if p.grad is not None])
            )

            optim.step()
            ema_model.update_parameters(mae._orig_mod if hasattr(mae, "_orig_mod") else mae)

            true_loss = loss.item()
            epoch_losses.append(true_loss)

        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        loss_history.append(avg_epoch_loss)
        pbar.set_postfix_str(f"Epoch {epoch} | Loss: {avg_epoch_loss:.10e}")

        if (epoch % vis_every == 0) and (x_vis_cache is not None):
            ema_model.module.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = ema_model.module(x_vis_cache)

                x_recon = out.x_recon[:4].detach().float().cpu()
                x_masked = out.x_masked[:4].detach().float().cpu()
                x_orig = x_vis_cache[:4].detach().float().cpu()

                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3,
                                      left=0.05, right=0.95, top=0.95, bottom=0.05)

                all_images = torch.cat([x_recon, x_masked, x_orig])
                vmin, vmax = all_images.min().item(), all_images.max().item()

                for j in range(4):
                    ax = fig.add_subplot(gs[0, j])
                    ax.imshow(x_recon[j].squeeze().numpy(), cmap="Blues", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Recon {j}", fontdict=title_font)
                    ax.axis("off")

                for j in range(4):
                    ax = fig.add_subplot(gs[1, j])
                    ax.imshow(x_masked[j].squeeze().numpy(), cmap="Blues", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Masked {j}", fontdict=title_font)
                    ax.axis("off")

                for j in range(4):
                    ax = fig.add_subplot(gs[2, j])
                    ax.imshow(x_orig[j].squeeze().numpy(), cmap="Blues", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Orig {j}", fontdict=title_font)
                    ax.axis("off")

                plt.suptitle(f"MAE Recon (EMA) - Epoch {epoch}", y=0.995)
                plt.savefig(f"samples/epoch_{epoch}.jpg", dpi=75, bbox_inches="tight", format="jpg")
                plt.close()


        last_ckpt = {
            "epoch": int(epoch),
            "model_state_dict": mae._orig_mod.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "best_loss": best_loss,
            "mask_ratio": mask_ratio
        }
        torch.save(last_ckpt, "checkpoints/last_mae.tar")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            last_ckpt["best_loss"] = best_loss
            torch.save(last_ckpt, "checkpoints/best_mae.tar")
