import os
import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

folder = "far"

for i in range(1, 5):
    image_name = f"img_0000{i}"
    for method in ['Cubic_Bspline', 'SLIIF']:  # 'Nearest', 'BiCubic_Bspline','Bilinear', 'Bicubic',
        test_file = f"{image_name}_tar.bmp_fftcc_icgn1_r16_deformation.txt"  #
        with open(os.path.join(
                rf"E:\dic\real_liif\{folder}\deformation\{method}\{test_file}"),
                encoding='utf-8') as f:
            test_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

        test_dic_res.resize(20, 20, 2)

        u = test_dic_res[:, :, 0]
        v = test_dic_res[:, :, 1]
        plt_range_u = 2
        plt_range_v = 10

        # 作图阶段
        fig = plt.figure()
        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(211)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(u, cmap=plt.cm.turbo)
        # 增加右侧的颜色刻度条
        plt.title(f"{method}-u")
        plt.colorbar(im)

        # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(212)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(v, cmap=plt.cm.turbo)
        # 增加右侧的颜色刻度条
        plt.colorbar(im)
        plt.title(f"{method}-v")

        plt.tight_layout()
        # show
        plt.show()
