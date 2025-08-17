import os
import warnings
from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

root_path = rf"..\image\crop\deformation"
gt_path = rf"..\image\crop\real_deform\crop"

test_mode = '10-小-1'

print("MAE:")
for method in ['Nearest', 'Bilinear', 'Bicubic', 'Cubic_Bspline','BiCubic_Bspline','LIIF','SLIIF']:  # ,, ,,, 'SLIIF_b', 'SLIIF_c',
    path = rf"{root_path}\{test_mode}\{method}"
    origin_list = []
    liif_list = []
    mode = test_mode.split("-")[1]
    if mode == '小':
        image_name = f"img_00001"
    elif mode == '中':
        image_name = f"img_00101"
    else:
        image_name = f"img_00201"

    dist = image_name + "_dis.npy"
    dis_file_name = image_name + "_tar.png_fftcc_icgn1_r16_deformation.txt"

    gt = np.load(rf'{gt_path}\{dist}')
    with open(os.path.join(path, dis_file_name), encoding='utf-8') as f:
        dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

    size = int(test_mode.split('-')[0])
    if size == 20:
        gt = gt[20:43:2, 20:43:2].reshape(144, 2)
    elif size == 15:
        gt = gt[15:49:2, 15:49:2].reshape(289, 2)
    else:
        gt = gt[10:54:2, 10:54:2].reshape(484, 2)


    rmse = mean_absolute_error(gt, dic_res)

    print(f"{method}：" + str(rmse))
