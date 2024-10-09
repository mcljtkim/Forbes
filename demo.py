import os
import numpy as np
import cv2
import torch
# from pattern_utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.utils import save_image

from brs import BRS
from adaface import AdaFace


def main(args):
    model = AdaFace(args).cuda()
    model.eval()
    
    optimizer = BRS(model, 'cuda', args)

    with torch.no_grad():
        img = cv2.imread(args.img_path)
        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        output = optimizer.optimize(img_torch)
        
        save_image(output[:, [2, 1, 0]] / 2 + 0.5, os.path.join(args.save_path, 'output.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--adaface_ckpt', type=str, default='./weights/adaface_ir101_ms1mv2.ckpt')
    parser.add_argument('--save_path', type=str, default='./output')
    args = parser.parse_args()

    args.transform_type = [
        'Mosaicking',
        'Horiz_mean',
        'Vertic_mean',
        'Averaging_blend',
        'Warping',
        'Sinusoid',
        'Checkerboard',
        'Speckle',
        'Noising_blend',
        'Scaling'
        ]
    args.transform_size = {
        'Warping': [1, 2, 7, 7],
        'Scaling': [7, 7, 1, 3, 1, 1],
        'Sinusoid': [7, 7, 3, 1, 1],
        'Speckle': [1, 3, 7, 7],
        'Averaging_blend': [3, 3, 7, 7],
        'Noising_blend': [3, 3, 7, 7]
    }
    args.transform_margin = {
        'Warping': 0.05,
        'Scaling': 1.05,
        'Speckle': 0.1
    }
    
    main(args)