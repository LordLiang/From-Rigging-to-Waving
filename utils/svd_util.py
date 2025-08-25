import functools
import importlib
import os
from functools import partial
from inspect import isfunction

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors

import decord
from einops import rearrange, repeat
from torchvision.transforms import Resize, Pad, InterpolationMode
from torch import nn

class SmoothAreaRandomDetection(object):
    
    def __init__(self, device="cuda", dtype=torch.float16):
        
        kernel_x = torch.zeros(3,3,3,3)
        for i in range(3):
            kernel_x[i,i,:,:] = torch.Tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.zeros(3,3,3,3)
        for i in range(3):
            kernel_y[i,i,:,:] = torch.Tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kernel_x = kernel_x.to(device, dtype)
        kernel_y = kernel_y.to(device, dtype)
        self.weight_x = kernel_x
        self.weight_y = kernel_y

        self.eps = 1/256.

    def detection(self, x, thr=0.0):
        original_dim = x.ndim
        if x.ndim > 4:
            b, f, c, h, w = x.shape
            x = rearrange(x, "b f c h w -> (b f) c h w")
        grad_xx = F.conv2d(x, self.weight_x, stride=1, padding=1)
        grad_yx = F.conv2d(x, self.weight_y, stride=1, padding=1)
        gradient_x = torch.abs(grad_xx) + torch.abs(grad_yx)
        gradient_x = torch.mean(gradient_x, dim=1, keepdim=True)
        gradient_x = repeat(gradient_x, "b 1 ... -> b 3 ...")
        if original_dim > 4:
            gradient_x = rearrange(gradient_x, "(b f) c h w -> b f c h w", b=b)
        return gradient_x <= thr 