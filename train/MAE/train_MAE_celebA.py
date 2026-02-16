import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA

import argparse
from torchvision.utils import save_image
from models.MAE import MAE

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
    parser.add_argument("--save_dir", type=str, default="mae-celeba")
    args = parser.parse_args()


    batch_size = 2048
    lr = 4e-3

    epochs = 1250
    num_workers = 16

    mask_ratio = 0.5        
    ema_decay = 0.9995      

    device = args.device

    transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    DATA_ROOT = Path("data").resolve()

    # Load CelebA dataset
    train_dataset = CelebA(
        root=str(DATA_ROOT), 
        split="train",
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=16,
        persistent_workers=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    mae = MAE(in_ch_pixel=3, unet_ch=128, unet_groups=32).to(device)
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

        for i, (x, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )

            if (epoch % vis_every == 0) and (i == 0):
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

                # only now move to CPU; this will sync once
                grids = torch.cat(
                    [
                        out.x_recon[:4].detach(),
                        out.x_masked[:4].detach(),
                        x_vis_cache[:4].detach(),
                    ],
                    dim=0,
                ).float().cpu().clamp(0.0, 1.0)

                save_image(grids, f"samples/epoch_{epoch}.jpg", nrow=4, padding=2)
 

        last_ckpt = {
            "epoch": int(epoch),
            "model_state_dict": mae._orig_mod.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "best_loss": best_loss,
            "mask_ratio": mask_ratio,
        }
        torch.save(last_ckpt, "checkpoints/last_mae.tar")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            last_ckpt["best_loss"] = best_loss
            torch.save(last_ckpt, "checkpoints/best_mae.tar")
