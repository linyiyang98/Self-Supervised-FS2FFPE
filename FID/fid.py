import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from numpy import random
from torchvision import models
from scipy import linalg
from data_loader import get_eval_loader
import cv2
import util as util

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        # inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def get_actvs(loader):
    actvs = []
    for x in tqdm(loader, total=len(loader)):
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/fid_test.png', util.RGB2BGR(util.tensor2numpy(util.denorm(x[0]) * 255)))
        actv = inception(x.to(device))
        actvs.append(actv)
    actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
    return actvs


@torch.no_grad()
def get_actvs_from_paths(paths, img_size=256, batch_size=50):
    print('Given paths %s and %s...' % (paths[0], paths[1]))
    actvs =[get_actvs(get_eval_loader(path, img_size, batch_size)) for path in paths]
    assert len(actvs) == 2
    return actvs


def calculate_kid(actvs, seed=42, max_subset_size=2048, num_subset=16):
    print('Calculating KID...')
    random.seed(seed)
    x, y = actvs
    n = x.shape[1]
    m = min(min(x.shape[0], y.shape[0]), max_subset_size)
    S = 0
    for _subset_idx in tqdm(range(num_subset), total=num_subset):
        xi = x[random.choice(x.shape[0], m, replace=False)]
        yi = y[random.choice(y.shape[0], m, replace=False)]
        a = sum((i @ i.T / n + 1) ** 3 for i in [xi, yi])
        b = (xi @ yi.T / n + 1) ** 3
        S += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid_value = S / num_subset / m
    return kid_value


def calculate_fid(actvs):
    print('Calculating FID...')
    mu, cov = [], []
    for i in actvs:
        mu.append(np.mean(i, axis=0))
        cov.append(np.cov(i, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='/data115_2/linyy/Xiangya-Jiazhuangxian/testA_results_vFFPE/',
                        help='paths to generated images')
    parser.add_argument('--paths', type=list, default=['/data115_2/linyy/Xiangya-Jiazhuangxian/testB/',\
                                                       '/data115_2/linyy/Xiangya-Jiazhuangxian/testA_results_vFFPE/'
                                                       ], help='paths to real and fake images(deprecated)')
    parser.add_argument('--img_size', type=int, default=224, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--gpu_ids', type=str, default='7', help='Set gpu mode; [cpu, cuda]')
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    inception = InceptionV3().eval().to(device)

    args.paths = [args.test_path, args.paths[0]]

    actvs = get_actvs_from_paths(args.paths, args.img_size, args.batch_size)
    KID = calculate_kid(actvs)
    FID = calculate_fid(actvs)
    print('KID * 100: ', KID * 100)
    print('FID: ', FID)
