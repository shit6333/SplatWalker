import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rl_utils import utils_drqv2 as drqutils

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        """ Assumes 84*84 input. """
        super().__init__()
        
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.repr_dim_tuple = (32, 35, 35)
        
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.apply(drqutils.weight_init)
    
    def forward(self, obs):
        # normalization is done in Encoder.forward(). Normalized from [0,255] to [-0.5, 0.5]
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)  # flatten
        return h
    

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)  # (4,4,4,4)
        x = F.pad(x, padding, 'replicate')  # replicate padding on each side by 4
        eps = 1.0 / (h + 2 * self.pad)  # 1 / len_padded_side
        arange = torch.linspace(-1.0 + eps,  # remove one pixel on each end
                                1.0 - eps,
                                h + 2 * self.pad,  # len_padded_side
                                device=x.device,
                                dtype=x.dtype)[:h]  # first h out of h + 2*pad
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        
        grid = base_grid + shift
        return F.grid_sample(x,  # grid_sample uses bilinear interpolation by default
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

    


if __name__ == "__main__":
    OBS_SHAPE  = (3, 84, 84)   # C, H, W
    BATCH_SIZE = 4
    PAD        = 4

    aug      = RandomShiftsAug(PAD)
    encoder  = Encoder(obs_shape=OBS_SHAPE)
    img_uint8 = torch.randint(
        0, 256, (BATCH_SIZE, *OBS_SHAPE), dtype=torch.uint8
    )


    img_aug   = aug(img_uint8.float())    # (B,3,84,84)
    feat      = encoder(img_aug)          # (B, 32*35*35)

    print("✓ RandomShiftsAug  output:", img_aug.shape)
    print("✓ Encoder repr dim :", feat.shape)   # (BATCH_SIZE, 39200)
