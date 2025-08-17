import os
from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

cubic_path = r"E:\dic\real_liif\far\deformation\Cubic_Bspline"
sliif_path = r"E:\dic\real_liif\far\deformation\SLIIF"

for i in range(1, 5):
    dis_file_name = f"img_0000{i}_tar.bmp_fftcc_icgn1_r16_deformation.txt"

    with open(os.path.join(cubic_path, dis_file_name), encoding='utf-8') as f:
        cubic_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

    with open(os.path.join(sliif_path, dis_file_name), encoding='utf-8') as f:
        sliif_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

    rmse = sqrt(mean_squared_error(cubic_dic_res, sliif_dic_res))
    mae = mean_absolute_error(cubic_dic_res, sliif_dic_res)

    print(f"Simple{i} rmse: {rmse}   mae: {mae}")
