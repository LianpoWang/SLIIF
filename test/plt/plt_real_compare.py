from matplotlib import pyplot as plt


def plt_print(rmse_list, mae_list):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体

    dev_x = [0, 25, 50, 75]

    plt.plot(dev_x, rmse_list, 'r', marker='o', markersize=10)
    plt.plot(dev_x, mae_list, 'b', marker='^', markersize=10)

    plt.xticks(dev_x, [0, 25, 50, 75])
    plt.xlabel("帧数")  # x轴标题
    plt.ylabel("ERROR")  # y轴标题

    # 绘制图例
    plt.legend(["RMSE", "MAE"])
    plt.show()
