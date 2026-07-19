import os
import cv2
import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim

data_root = 'data/MixUpsample_Dataset_R2A2_256/test'
h_folder = os.path.join(data_root, 'H')
l_folder = os.path.join(data_root, 'L')

h_names = sorted(os.listdir(h_folder))

psnr_list = []
ssim_list = []

for h_name in h_names:
    l_name = h_name.replace('_H_', '_L_')
    h_path = os.path.join(h_folder, h_name)
    l_path = os.path.join(l_folder, l_name)

    img_h = cv2.imread(h_path)
    img_l = cv2.imread(l_path)

    psnr_val = calculate_psnr(img_h, img_l, crop_border=0)
    ssim_val = calculate_ssim(img_h, img_l, crop_border=0)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
print(f'Baseline (L vs H):  average PSNR = {avg_psnr:.4f},  average SSIM = {avg_ssim:.4f}')
