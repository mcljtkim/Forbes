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

class LFW(data.Dataset):
    def __init__(self, args):
        self.dataset_root = args.dataset_root
        self.pair_list = np.load(os.path.join(args.dataset_root, 'lfw', 'lfw.npy'))
        self.issame = np.load(os.path.join(args.dataset_root, 'lfw', 'lfw_list.npy'))
    
    def __getitem__(self, index):
        splits = self.pair_list[index]

        im1 = (splits[0] + 1) / 2
        im2 = (splits[1] + 1) / 2

        tf = self.issame[index] * 1

        return {'im1': im1, 'im2': im2, 'tf': tf}

    def __len__(self):
        return len(self.pair_list)


def main(args):
    model = AdaFace(args).cuda()
    model.eval()
    
    optimizer = BRS(model, 'cuda', args)

    test_set = LFW(args)
    test_data_loader = DataLoader(dataset=test_set, num_workers=12, batch_size=args.batch_size, shuffle=False)

    labels = []
    sims = []
    with torch.no_grad():
        for iteration, image_blob in enumerate(test_data_loader, 1):
            for key in ['im1', 'im2']:
                image_blob[key] = image_blob[key].cuda()
            b = image_blob['im1'].shape[0]
            img_inp, img_ref = map(lambda t: t * 2 - 1, (image_blob['im1'], image_blob['im2']))

            optimized_img = optimizer.optimize(img_inp)

            feat, _ = model(torch.cat((optimized_img, img_ref), dim=0))
            
            sim = torch.bmm(feat[:b].unsqueeze(1), feat[b:].unsqueeze(2)).squeeze(1).squeeze(1)
            sims.extend(sim.cpu())
            
            labels.extend(image_blob['tf'])

            print(f'===> Validation [{(iteration)}/{len(test_data_loader)}] Test proceeding...', end='\r')
        acc, th = cal_accuracy(sims, labels)
    print('============== [Forbes]  Accuracy : %.6f, th: %.6f ==============' % (acc, th / len(test_data_loader)))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--adaface_ckpt', type=str, default='./weights/adaface_ir101_ms1mv2.ckpt')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    
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
