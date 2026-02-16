import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)


def make_skip_connection(dim_in, dim_out):
    if dim_in == dim_out:
        return nn.Identity()
    return nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)


def make_block(dim_in, dim_out, num_groups, dropout=0):
    return nn.Sequential(nn.GroupNorm(num_groups=num_groups, num_channels=dim_in), 
                         nn.SiLU(),
                         nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
                         nn.Conv2d(dim_in, dim_out, 3, 1, 1))


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_groups=32, dropout=0.1, attn=False):
        super().__init__()

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return h
