import os
import torch
import cv2
import numpy as np
import math
import random
import torch.nn.functional as F
import torch.nn as nn

def Mosaicking(x, alpha):
    b, c, h, w = x.size()
    y = torch.stack(torch.stack(x.chunk(7, dim=2)).chunk(7, dim=4)).mean(dim=(-1, -2)).permute(2, 3, 1, 0)
    y = F.interpolate(y, (h, w), mode='nearest')
    return y

def Horiz_mean(x, param=None):
    b, c, h, w = x.size()

    return torch.mean(x, dim=3, keepdim=True).expand(b, c, h, w)

def Vertic_mean(x, param=None):
    b, c, h, w = x.size()

    return torch.mean(x, dim=2, keepdim=True).expand(b, c, h, w)

def Warping(x, flo):
    B, C, H, W = x.size()
    _, _, h, w = flo.size()

    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

    grid = torch.cat((xx, yy), 1).float()

    flo = F.interpolate(flo, (H, W), mode='bilinear', align_corners=True)
    flo[:, 0] = flo[:, 0] * H / h
    flo[:, 1] = flo[:, 1] * W / w

    if x.is_cuda:
        grid = grid.to(x.device)

    vgrid = torch.autograd.Variable(grid) + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)

    return output

def Sinusoid(x, param=None):
    b, c, h, w = x.size()
    wp, hp, _, _, _, _ = param.size()

    w_grid = w // wp
    h_grid = h // hp

    yy = torch.linspace(-np.pi, np.pi, h_grid).view(1, 1, 1, 1, 1, h_grid, 1).expand(wp, hp, 1, 3, 1, h_grid, w_grid)
    xx = torch.linspace(-np.pi, np.pi, w_grid).view(1, 1, 1, 1, 1, 1, w_grid).expand(wp, hp, 1, 3, 1, h_grid, w_grid)

    grid = torch.cat((xx, yy), 4).float().to(x.device)

    alpha = param
    grid = grid[:, :, :, :, 0] * alpha + grid[:, :, :, :, 1] * (1 - alpha)

    sin2d = 0.2 * torch.sin(grid * 4)
    
    return sin2d.permute(2, 3, 1, 4, 0, 5).reshape(b, c, h, w)

def Checkerboard(x, param=None):
    b, c, h, w = x.size()
    param = torch.ones(7, 7, 1, 3, 1, 1).to(x.device)
    xx = torch.arange(0, 4).view(1, 1, 1, 4).expand(1, 1, 4, 4)
    yy = torch.arange(0, 4).view(1, 1, 4, 1).expand(1, 1, 4, 4)
    grid = (xx + yy) % 2
    grid = grid.float() - 0.5
    block_size = h // 7
    grid = F.interpolate(grid, (block_size, block_size), mode='nearest').to(x.device)

    return (param * grid).permute(2, 3, 1, 4, 0, 5).reshape(1, c, h, w).expand(b, c, h, w) * 0.3

def Speckle(x, param=None):
    b, c, h, w = x.size()
    alpha = F.interpolate(param, (h, w), mode='bilinear')

    return alpha

def Scaling(x, input_color):
    b, c, h, w = x.size()
    blockh = input_color.size()[1]
    blockw = input_color.size()[0]
    x = torch.stack(torch.stack(x.chunk(blockh, dim=2)).chunk(blockw, dim=4))
    x = torch.clamp(((x+1) * input_color / 2) * 2 - 1, min=-1, max=1)
    x = x.permute(2, 3, 1, 4, 0, 5).reshape(b, c, h, w)
    
    return x

def generate_zto(size, device):
    init_param = torch.rand(np.prod(size), device=device).float()
    bound = np.array([(0, 1)] * np.prod(size))

    return init_param, bound

def generate_sym(size, device, value):
    prob = torch.rand(np.prod(size), device=device).float()
    init_param = torch.zeros_like(prob, device=device).float()
    _len = (prob > 0.5).sum().item()
    _min = 1.0/value
    init_param[prob > 0.5] = torch.rand(_len, device=device).float() * (value - 1) + 1
    init_param[prob <= 0.5] = torch.rand(np.prod(size) - _len, device=device).float() * (1 - _min) + _min
    bound = np.array([(_min, value)] * np.prod(size))

    return init_param, bound

def generate_minmax(size, device, min, max):
    init_param = torch.rand(np.prod(size), device=device).float() * (max - min) + min

    return init_param, np.array([(min, max)] * np.prod(size))

def generate_warp(size, device, min, max):
    init_param = torch.rand(np.prod(size), device=device).float() * (max - min) + min
    init_param = init_param.view(size)
    init_param[:, :, [0, -1], :] = 0
    init_param[:, :, :, [0, -1]] = 0
    init_param = init_param.view(-1)
    bound = np.array([(min, max)] * np.prod(size))
    bound[(init_param == 0).cpu()] = 0
    
    return init_param, bound

def generate_blend(size, device):
    init_param = torch.rand(np.prod(size), device=device).float()
    bound = np.array([(None, None)] * np.prod(size))
    
    return init_param, bound

def generate_param(transform, size, device):
    if transform == 'Warping':
        param, bound = generate_warp(size, device, -0.3, 0.3)
    elif transform == 'Scaling':
        param, bound = generate_sym(size, device, 1.1)
    elif transform == 'Sinusoid':
        param, bound = generate_zto(size, device)
    elif transform == 'Speckle':
        param, bound = generate_minmax(size, device, -0.5, 0.5)
    elif 'blend' in transform:
        param, bound = generate_blend(size, device)
        
    return param, bound
