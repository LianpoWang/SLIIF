import ast
import logging
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

from demo import ModelTest

app = FastAPI()

model_test = ModelTest()

curr_iter_list = []
all_iter_list = []

curr_mode = "LIIF_img_00001_tar.png"

# 初始化日志配置
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pixel_calculation.log", encoding='utf-8'),  # 日志保存到文件
        logging.StreamHandler()  # 同时在控制台输出
    ]
)
logger = logging.getLogger(__name__)


@app.get("/")
def read_root():
    return {"Hello": "World"}


# 新增全局变量用于跟踪上一次的模型名称和图像路径
prev_model_name = None
prev_image_path = None


@app.get("/get_pixel/{image_path}/{model_name}/{location}/{iter}")
async def get_pixel_cord(image_path: str, model_name: str, location: str, iter: int):
    global prev_model_name, prev_image_path  # 声明使用全局变量
    location_array = ast.literal_eval(location)
    result = []
    start = time.time()

    res = np.array(location_array).astype(np.float32)
    res = torch.Tensor(res)
    res = res.view(-1, res.shape[-1])

    result = model_test.get_pixel(image_path, res, model_name)

    end = time.time()
    use_time = round(((end - start) * 1000), 2)

    # 处理上一批次的累计数据
    if iter == 1 and len(curr_iter_list) > 0:
        all_iter_list.append(sum(curr_iter_list))
        curr_iter_list.clear()

    # 检查当前模型和图像路径是否与上一次不同
    if model_name != prev_model_name or image_path != prev_image_path:
        # 计算并记录上一批次的平均值（排除初始值）
        if len(all_iter_list) > 1:
            temp_list = all_iter_list[1:]
            all_iter_list.clear()
            all_mean = sum(temp_list) / len(temp_list)
            # 使用上一批次的模型和图像路径生成标识
            prev_temp_mode = f"{prev_model_name}_{prev_image_path}" if prev_model_name and prev_image_path else ""
            if prev_temp_mode:  # 避免首次请求时的空值情况
                logger.warning(f"{prev_temp_mode}计算时间:{all_mean}毫秒")

    # 更新跟踪的参数为当前值
    prev_model_name = model_name
    prev_image_path = image_path

    # 累计当前请求的时间
    curr_iter_list.append(use_time)

    # 处理返回结果
    resStr = ""
    for h in result:
        preds = ",".join([str(item) for item in h])
        preds += "@"
        resStr += preds

    return resStr


if __name__ == '__main__':
    # 运行fastapi程序
    uvicorn.run(app="main:app", host="127.0.0.1", port=8000)
