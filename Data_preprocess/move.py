import shutil, os

if __name__ == "__main__":
    dir0 = '/data115_1/linyy/TCGA-BRCA/TCGA-BRCA-10/FS2FFPE-Norm10/trainA/'
    dir = os.listdir(dir0)

    loc = '/data115_1/linyy/TCGA-BRCA/TCGA-BRCA-10/FS2FFPE-Norm10/testA/'

    for i in range(len(dir)):
        if i % 10==0:
            shutil.move(os.path.join('/data115_1/linyy/TCGA-BRCA/TCGA-BRCA-10/FS2FFPE-Norm10/trainA/',dir[i]), loc)
