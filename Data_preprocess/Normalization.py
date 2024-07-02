import torchstain
import os
from multiprocessing import Pool,Process
from PIL import Image
import numpy as np
import cv2
import random
from torchvision import transforms


target = cv2.cvtColor(cv2.imread("/data112/linyy/ST-net_results/448_20_moban3.png"), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

def transform(dir):
    target = cv2.cvtColor(cv2.imread("/data112/linyy/ST-net_results/448_20_moban3.png"), cv2.COLOR_BGR2RGB)
    torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(T(target))
    print(dir)
    for root, _, fnames in os.walk(dir):
        random.shuffle(fnames)
        if not os.path.exists(root.replace('tiles', 'tiles_Norm')): os.makedirs(root.replace('tiles', 'tiles_Norm'))
        for id, fname in enumerate(fnames):
            path = os.path.join(root, fname)
            to_transform = np.uint8(Image.open(path).convert('RGB'))
            t_to_transform = T(to_transform)
            norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)
            cv2.imwrite(path.replace('tiles', 'tiles_Norm'), cv2.cvtColor(np.uint8(np.array(norm)), cv2.COLOR_RGB2BGR))
            if id >= 100 - 1: break
    print(8888888888888888888888888888888888888)


if __name__ == "__main__":
    dir0 = '/data112/linyy/TCGA_BRCA/FFPE_448_20/tiles/'
    dir = os.listdir(dir0)

    pool = Pool(processes=1)
    for i in range(len(dir)):
        pool.apply_async(transform,args=(os.path.join('/data112/linyy/TCGA_BRCA/FFPE_448_20/tiles/',dir[i]),))
    pool.close()
    pool.join()