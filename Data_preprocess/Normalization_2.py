import glob
from multiprocessing import Pool,Process
import torch
from torchvision import transforms
import torchstain
import cv2
import numpy as np

target = cv2.cvtColor(cv2.imread("/data112/linyy/ST-net_results/448_20_moban_2.png"), cv2.COLOR_BGR2RGB)
dir = glob.glob('/data112/linyy/TCGA_BRCA/FFPE_448_20/tiles/*/*.png')

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
torch_normalizer.fit(T(target))

for id, img in enumerate(dir):
    to_transform = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)
    norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

    print(norm)

    cv2.imwrite(img.replace('tiles', 'tiles_Norm'), cv2.cvtColor(np.uint8(np.array(norm)), cv2.COLOR_RGB2BGR))