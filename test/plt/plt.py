import os

import cv2
from matplotlib import pyplot as plt


def plt_print(image_dir, plt_range=50):
    for index in range(1, 5):
        image_name = f"img_00{index - 1}01_ref.png"

        for method in ["INTER_CUBIC", "SRGAN", "EDSR", "RDN", "Speckle-SRGAN"]:
            groudtruth_path = rf'D:\learn\AI_learn\SR\datasets\DIV2K\DIV2K_test\{index}'
            test_image_path = rf"{image_dir}\sr\{method}"

            groudtruth = cv2.imread(os.path.join(groudtruth_path, image_name), cv2.IMREAD_GRAYSCALE)
            test_image = cv2.imread(os.path.join(test_image_path, image_name), cv2.IMREAD_GRAYSCALE)

            error_image = cv2.absdiff(groudtruth, test_image)

            # cv2.imshow(f'{method}groudtruth', groudtruth)
            # cv2.imshow(f'{method}test_image', test_image)
            # cv2.imshow(f'{method}error_image', error_image)
            #

            # 作图阶段
            fig = plt.figure()
            # 定义画布为1*1个划分，并在第1个位置上进行作图
            ax = fig.add_subplot(111)
            # 作图并选择热图的颜色填充风格，这里选择hot
            im = ax.imshow(error_image, cmap=plt.cm.turbo, vmax=plt_range, vmin=-plt_range)
            # 增加右侧的颜色刻度条
            plt.colorbar(im)
            plt.title(method + str(index))
            plt.show()
