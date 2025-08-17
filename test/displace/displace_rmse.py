import os
import warnings
from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

root_path = rf"..\image\crop\deformation"
gt_path = rf"..\image\crop\real_deform\crop"

test_mode = '20-中-1'

print("RMSE:")
for method in ['Nearest', 'Bilinear', 'Bicubic', 'Cubic_Bspline',
               'BiCubic_Bspline','LIIF','SLIIF']:  # 'Nearest', 'Bilinear', 'Bicubic', 'Cubic_Bspline',,,, 'LIIF', 'SLIIF_b', 'SLIIF_c',
    path = rf"{root_path}\{test_mode}\{method}"
    origin_list = []
    liif_list = []
    mode = test_mode.split("-")[1]

    rmse_list = []

    for i in range(1, 6):
        if mode == '小':
            image_name = f"img_0000{i}"
        elif mode == '中':
            image_name = f"img_0010{i}"
        else:
            image_name = f"img_0020{i}"

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

        rmse = sqrt(mean_squared_error(gt, dic_res))
        rmse_list.append(rmse)

    print(f"{method}：" + str(np.mean(rmse_list)))
    # print(rmse_list)
