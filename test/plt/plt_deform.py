import os
import warnings

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "font.size": 17
})

warnings.filterwarnings("ignore")

folder = "20-中-1"

scaler = 15

size = int(folder.split('-')[0])
if size == 20:
    scaler = 24
    # scaler = 12
elif size == 15:
    scaler = 17
else:
    scaler = 22

is_error_map = False

gt = False

mode = folder.split("-")[1]
if mode == '小':
    image_name = f"img_00002"
elif mode == '中':
    image_name = f"img_00101"
else:
    image_name = f"img_00201"

for method in ['SLIIF']:  #
    test_file = f"{image_name}_tar.png_fftcc_icgn1_r16_deformation.txt"  #
    with open(os.path.join(
            rf"..\image\crop\deformation\{folder}\{method}\{test_file}"),
            encoding='utf-8') as f:
        test_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

    groudtruth_file = f"{image_name}_dis.npy"
    groudtruth_dic_res = np.load(
        rf'..\image\crop\real_deform\crop\{groudtruth_file}')[
                         size:64 - size:1, size:64 - size:1]

    x = np.arange(0, 17)
    y = np.arange(0, 17)

    test_dic_res.resize(scaler, scaler, 2)

    if is_error_map:
        dic_res = np.subtract(test_dic_res, groudtruth_dic_res)
        u = dic_res[:, :, 0]
        v = dic_res[:, :, 1]
        plt_range = 0.05
    elif gt:
        u = groudtruth_dic_res[:, :, 0]
        v = groudtruth_dic_res[:, :, 1]
        method = "Ground Truth"
    else:
        u = test_dic_res[:, :, 0]
        v = test_dic_res[:, :, 1]
        plt_range_u = 5
        plt_range_v = 10

    if is_error_map:
        # plt.rcParams.update({"font.size": 13})
        # 作图阶段
        fig = plt.figure()
        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(211)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(u, cmap=plt.cm.turbo, vmax=plt_range, vmin=-plt_range)
        # 增加右侧的颜色刻度条
        # plt.colorbar(im)
        plt.title(f"{method}-error-u",y=1.1)

        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(212)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(v, cmap=plt.cm.turbo, vmax=plt_range, vmin=-plt_range)
        # 增加右侧的颜色刻度条
        # plt.colorbar(im)
        plt.title(f"{method}-error-v",y=1.1)

        plt.tight_layout()
        # show
        plt.show()
    else:
        # 作图阶段
        fig = plt.figure()
        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(211)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(u, cmap=plt.cm.turbo)
        # 增加右侧的颜色刻度条
        plt.title(f"{method}-u",y=1.1)
        # plt.colorbar(im)

        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(212)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(v, cmap=plt.cm.turbo)
        # 增加右侧的颜色刻度条
        # plt.colorbar(im)
        plt.title(f"{method}-v",y=1.1)

        plt.tight_layout()
        # show
        plt.show()

    if method == "gt":
        break