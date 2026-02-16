from typing import List, NamedTuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import (
    Downsample,
    Upsample,
    ResBlock,
)


class MAEOutput(NamedTuple):
    x_masked: torch.Tensor
    mask: torch.Tensor
    x_recon: torch.Tensor

class MAEFeatures(NamedTuple):
    LocationWiseFeatures: List[torch.Tensor]  # List[[B,C_i,H_i,W_i]]
    PooledFeatures: List[torch.Tensor]        # List[[B,Npool_i,C_i]]

def _patch_stats_vectors(f_map: torch.Tensor, patch: int, eps: float = 1e-6) -> torch.Tensor:
    """
    f_map: [B, C, H, W]
    returns: [B, 2 * (H/patch * W/patch), C]
    """
    B, C, H, W = f_map.shape
    assert H % patch == 0 and W % patch == 0, "H,W must be divisible by patch"

    mean = F.avg_pool2d(f_map, kernel_size=patch, stride=patch)          # [B,C,H/p,W/p]
    mean2 = F.avg_pool2d(f_map * f_map, kernel_size=patch, stride=patch) # [B,C,H/p,W/p]
    var = (mean2 - mean * mean).clamp_min(0.0)
    std = torch.sqrt(var + eps)

    mean_vecs = mean.flatten(2).transpose(1, 2)  # [B,P,C]
    std_vecs  = std.flatten(2).transpose(1, 2)   # [B,P,C]
    return torch.cat([mean_vecs, std_vecs], dim=1)  # [B,2P,C]


def pooled_vectors_from_map(f_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pooled (non-location-wise) vectors only:
      - global mean, global std (2 vectors)
      - 2x2 patch mean/std
      - 4x4 patch mean/std
    Output: [B, Npool, C]
    """
    B, C, H, W = f_map.shape

    g_mean = f_map.mean(dim=(2, 3))
    g_var  = f_map.var(dim=(2, 3), unbiased=False)
    g_std  = torch.sqrt(g_var + eps)
    global_vecs = torch.stack([g_mean, g_std], dim=1)  # [B,2,C]

    pooled: List[torch.Tensor] = [global_vecs]
    if H >= 2 and W >= 2 and (H % 2 == 0) and (W % 2 == 0):
        pooled.append(_patch_stats_vectors(f_map, patch=2, eps=eps))
    if H >= 4 and W >= 4 and (H % 4 == 0) and (W % 4 == 0):
        pooled.append(_patch_stats_vectors(f_map, patch=4, eps=eps))

    return torch.cat(pooled, dim=1)


class Unet(nn.Module):
    def __init__(
            self,
            # circs [in_ch = 1*4*4, ch = 1*4*4, groups=16]
            # celeb [in_ch = 3*4*4, ch=128, groups=32]
            in_ch: int, # given as a function of pixel space channels and patchify_size
            ch: Union[int, None], # refactor later
            groups: int = 16,
            ch_mul: List[int] = [1, 2, 4, 8],
            # be more aggressive in lower dimension case 16 -> 8 -> 4 -> 2
            # default 32 -> 16 -> 8 -> 4
            spatial_downsample: List[int] = [2, 2, 2],
            dropout: float = 0.1,
        ):
        super().__init__()

        self.in_ch = in_ch
        self.ch = in_ch if ch is None else ch
        self.ch_mul = ch_mul
        # 32 x 32 x (4*4*1) on ch1
        # 32 x 32 x (8*8*3) on ch3
        self.groups = groups
        self.dropout = dropout

        self.input_proj = nn.Conv2d(in_ch, self.ch, 3, 1, 1)

        num_res = len(self.ch_mul)
        assert len(spatial_downsample) == max(num_res - 1, 0), (
            f"spatial_downsample must have {max(num_res - 1, 0)} entries for {num_res} resolutions"
        )
        self.spatial_downsample = spatial_downsample

        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])

        self.make_paths()

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=2 * self.ch),
            nn.SiLU(),
            nn.Conv2d(2 * self.ch, self.in_ch, 3, 1, 1),
        )

    def forward_encoder(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        h = self.input_proj(x) 
        encoder_features: List[torch.Tensor] = []
        for block_group in self.down:
            h = block_group[0](h)
            h = block_group[1](h)
            encoder_features.append(h)
            if len(block_group) > 2:
                h = block_group[2](h)

        return encoder_features


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # skip connection
        input_proj = self.input_proj(x)
        h = input_proj

        down_path = []
        encoder_features = []

        for block_group in self.down:
            h = block_group[0](h)
            h = block_group[1](h)
            down_path.append(h)
            encoder_features.append(h)

            if len(block_group) > 2:
                h = block_group[2](h)

        h = self.mid[0](h)
        h = self.mid[1](h)

        for block_group in self.up:
            h = torch.cat((h, down_path.pop()), dim=1)
            h = block_group[0](h)
            h = block_group[1](h)

            if len(block_group) > 2:
                h = block_group[2](h)

        x = torch.cat((h, input_proj), dim=1)
        output = self.final(x)

        return output

    def _down_stride(self, res: int) -> int:
        if res >= (len(self.ch_mul) - 1):
            return 1
        return self.spatial_downsample[res]

    def _up_scale(self, res: int) -> int:
        if res == 0:
            return 1
        return self.spatial_downsample[res - 1]

    def make_transition(self, res, down):
        dim = self.ch * self.ch_mul[res]

        if down:
            is_last_res = (res == (len(self.ch_mul) - 1))
            if is_last_res:
                return Downsample(dim, dim, stride=1)

            dim_out = self.ch * self.ch_mul[res + 1]
            stride = self._down_stride(res)
            return Downsample(dim, dim_out, stride=stride)

        is_first_res = (res == 0)
        if is_first_res:
            return Upsample(dim, dim, scale_factor=1)

        dim_out = self.ch * self.ch_mul[res - 1]
        scale = self._up_scale(res)
        return Upsample(dim, dim_out, scale_factor=scale)

    def make_res(self, res, down):
        dim = self.ch * self.ch_mul[res]
        transition = self.make_transition(res, down)

        if down:
            block1 = ResBlock(dim, dim, self.groups, self.dropout)
            block2 = ResBlock(dim, dim, self.groups, self.dropout)
        else:
            block1 = ResBlock(2 * dim, dim, self.groups, self.dropout)
            block2 = ResBlock(dim, dim, self.groups, self.dropout)

        return nn.ModuleList([block1, block2, transition])

    def make_paths(self):
        num_res = len(self.ch_mul)

        for res in range(num_res):
            is_last_res = (res == (num_res - 1))

            down_blocks = self.make_res(res, down=True)
            up_blocks = self.make_res(res, down=False)

            if is_last_res:
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]

            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)

        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList([
            ResBlock(nch, nch, self.groups, self.dropout),
            ResBlock(nch, nch, self.groups, self.dropout),
        ])

# ==========================================================
# For 128x128:
#   choose patchify_size=4  => grid is 32x32
# For 256x256 (paper):
#   choose patchify_size=8  => grid is 32x32
# ==========================================================
def patchify_to_grid(x: torch.Tensor, patchify_size: int) -> torch.Tensor:
    """
    x: [B, C, H, W]
    returns grid: [B, C*(p*p), H/p, W/p] where p=patchify_size
    """
    B, C, H, W = x.shape
    p = patchify_size
    assert H % p == 0 and W % p == 0, "H,W must be divisible by patchify_size"
    gh, gw = H // p, W // p

    # [B, C, gh, p, gw, p] -> [B, gh, gw, C, p, p] -> [B, gh, gw, C*p*p] -> [B, C*p*p, gh, gw]
    x = x.view(B, C, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(B, gh, gw, C * p * p)
    grid = x.permute(0, 3, 1, 2).contiguous()
    return grid


def unpatchify_from_grid(grid: torch.Tensor, patchify_size: int, out_channels: int) -> torch.Tensor:
    """
    grid: [B, C*(p*p), gh, gw]
    returns x: [B, C, gh*p, gw*p]
    """
    B, Cp2, gh, gw = grid.shape
    p = patchify_size
    C = out_channels
    assert Cp2 == C * p * p, f"grid channels must be C*p*p = {C*p*p}, got {Cp2}"

    # [B, C*p*p, gh, gw] -> [B, gh, gw, C*p*p] -> [B, gh, gw, C, p, p] -> [B, C, gh, p, gw, p] -> [B, C, H, W]
    x = grid.permute(0, 2, 3, 1).contiguous()
    x = x.view(B, gh, gw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, gh * p, gw * p)
    return x


class MAE(nn.Module):
    def __init__(
        self,
        in_ch_pixel = 3,
        unet_ch = None,
        unet_groups = 16,
        mask_ratio: float = 0.5,
        patchify_size: int = 4,   # 128->32 grid uses 4; 256->32 uses 8
        mask_block: int = 2,      # 2x2 masking on the grid
        feature_eps: float = 1e-6,
        spatial_downsample: List[int] = [2, 2, 2],
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.patchify_size = int(patchify_size)
        self.mask_block = int(mask_block)
        self.feature_eps = float(feature_eps)

        self.unet = Unet(in_ch=in_ch_pixel * patchify_size * patchify_size,
        ch=unet_ch, groups=unet_groups, spatial_downsample=spatial_downsample)

    def _make_grid_mask(self, grid: torch.Tensor) -> torch.Tensor:
        """
        grid: [B, D, gh, gw]
        returns mask_grid: [B, 1, gh, gw] bool
        Mask is sampled on a (gh/m, gw/m) coarse grid of m x m blocks, then upsampled by repeat.
        """
        B, _, gh, gw = grid.shape
        m = self.mask_block
        assert gh % m == 0 and gw % m == 0, "grid H,W must be divisible by mask_block"

        bh, bw = gh // m, gw // m
        # Bernoulli per block
        mask_block = (torch.rand((B, 1, bh, bw), device=grid.device) < self.mask_ratio)  # bool
        # expand each block to m x m in grid coords
        mask_grid = mask_block.repeat_interleave(m, dim=2).repeat_interleave(m, dim=3)
        return mask_grid

    def _apply_grid_mask(self, grid: torch.Tensor, mask_grid: torch.Tensor) -> torch.Tensor:
        """
        grid: [B, D, gh, gw]
        mask_grid: [B, 1, gh, gw] bool
        """
        # broadcast mask over channel dim
        return grid.masked_fill(mask_grid, 0.0)

    def _get_features(self, x: torch.Tensor) -> MAEFeatures:
        """
        x: [B,C,H,W] (unmasked, pixel space)
        returns:
        - LocationWiseFeatures: List[[B, C_i, H_i, W_i]] from encoder *on the patchified grid*
        - PooledFeatures:       List[[B, Npool_i, C_i]] pooled from those feature maps
            plus one extra pooled vector from the *grid input* (mean of squares per channel).
        """
        assert x.dim() == 4

        self.unet.eval()

        p = self.patchify_size

        # 1) Patchify pixels -> grid space (paper pixel-space mode)
        #    grid: [B, C*p*p, H/p, W/p]
        grid = patchify_to_grid(x, patchify_size=p)

        # 2) Encoder runs on grid 
        loc_maps: List[torch.Tensor] = self.unet.forward_encoder(grid)  # List[[B, C_i, H_i, W_i]]

        # 3) Pooled features per scale (same logic as before)
        pooled_vecs: List[torch.Tensor] = [
            pooled_vectors_from_map(f, eps=self.feature_eps) for f in loc_maps
        ]

        # 4) Extra pooled feature: mean-of-squares over spatial dims of the *grid input*
        #    shape: [B, 1, C0] where C0 = C*p*p
        pooled_vecs.append((grid * grid).mean(dim=(2, 3)).unsqueeze(1))

        return MAEFeatures(LocationWiseFeatures=loc_maps, PooledFeatures=pooled_vecs)
 
    def forward(self, x: torch.Tensor) -> MAEOutput:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, C, H, W = x.shape
        p = self.patchify_size

        grid = patchify_to_grid(x, patchify_size=p)          # [B, C*p*p, H/p, W/p]
        mask_grid = self._make_grid_mask(grid)               # [B,1,gh,gw] bool
        grid_masked = self._apply_grid_mask(grid, mask_grid) # [B, C*p*p, gh, gw]

        grid_recon = self.unet(grid_masked)                  # [B, C*p*p, gh, gw]

        x_masked = unpatchify_from_grid(grid_masked, patchify_size=p, out_channels=C)
        x_recon  = unpatchify_from_grid(grid_recon,  patchify_size=p, out_channels=C)

        mask_pixel = mask_grid.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)

        return MAEOutput(x_masked=x_masked, mask=mask_pixel, x_recon=x_recon) 

