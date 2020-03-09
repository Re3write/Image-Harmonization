from PIL import Image
import numpy as np
import os
from skimage import data, img_as_float
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr

IMAGE_SIZE = np.array([256, 256])


# comp_name = 'data/a0002_1_4.jpg'


def real_name(name):
    name_parts = name.split('_')
    real_name = name.replace(('_' + name_parts[-2] + '_' + name_parts[-1]), '.jpg')
    return real_name


def mask_name(name):
    name_parts = name.split('_')
    mask_name = name.replace(('_' + name_parts[-1]), '.png')
    return mask_name


def evaluation(root_dir):
    mse_total = 0
    psnr_total = 0
    fmse_total = 0
    count = 0
    for comp_name in os.listdir(root_dir):
        comp = Image.open(comp_name)
        comp = comp.resize(IMAGE_SIZE, Image.BICUBIC)
        comp = np.array(comp, dtype=np.float32)

        real = Image.open(real_name(comp_name))
        real = real.resize(IMAGE_SIZE, Image.BICUBIC)
        real = np.array(real, dtype=np.float32)

        mask = Image.open(mask_name(comp_name))
        mask = mask.convert('1')
        mask = mask.resize(IMAGE_SIZE, Image.BICUBIC)
        mask = np.array(mask, dtype=np.uint8)
        fore_area = np.sum(np.sum(mask, axis=0), axis=0)
        mask = mask[..., np.newaxis]

        mse_total += mse(comp, real)
        psnr_total += psnr(real, comp, data_range=comp.max() - comp.min())
        fmse_total += mse(comp * mask, real * mask) * 256 * 256 / fore_area
        count += 1
    print("%s MSE %0.2f | PSNR %0.2f | fMSE %0.2f" % (
        comp_name, mse_total / count, psnr_total / count, fmse_total / count))


if __name__ == '__main__':
    root_dir = ''
    evaluation(root_dir)
