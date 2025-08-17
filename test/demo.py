import argparse
import math
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch import tensor
from torchvision import transforms
from tqdm import tqdm

import models
from test import batched_predict
from utils import make_coord_new

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./image/img_00301_ref.png')
parser.add_argument('--model', default='E:/dic/train/liif/epoch-last.pth')
parser.add_argument('--resolution', default='5000,5000')
parser.add_argument('--output', default='output.png')
parser.add_argument('--gpu', default=0)
args = parser.parse_args()


class ModelTest:
    def __init__(self):
        self.currModel = None
        self.model = None

    def get_pixel(self, img_path, tran, model_name, n=64):
        # 加载模型
        if self.currModel != model_name:
            model_path = f'E:/dic/train/{model_name}/epoch-last.pth'
            self.model = models.make(
                torch.load(model_path, map_location=torch.device('cuda:' + str(args.gpu)))['model'],
                load_sd=True).cuda(args.gpu)
            self.currModel = model_name

        if n == 64:
            img_path = f'./image/crop/{img_path}'
            # img_path = f'../dft/dft_res_120/{img_path}'
        else:
            img_path = f'E:/dic/real_liif/far/{img_path}'
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        coord = make_coord_new(tran, n).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / n
        cell[:, 1] *= 2 / n

        pred = batched_predict(self.model, ((img - 0.5) / 0.5).cuda(args.gpu).unsqueeze(0),
                               coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        n = int(math.sqrt(len(tran)))
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(n, n, 3).permute(2, 0, 1).cpu()

        res = torch.mean(pred, dim=0)

        res = res.numpy()

        return res * 255.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='img_00001_ref.png')
    parser.add_argument('--model', default='./save/_train_rdn-liif/epoch-best.pth')
    parser.add_argument('--resolution', default='64,64')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default="0")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model, map_location=torch.device('cuda:' + str(args.gpu)))['model'],
                        load_sd=True).cuda(args.gpu)

    h, w = list(map(int, args.resolution.split(',')))
    res = []
    test = ModelTest()
    for x in tqdm(range(0, h)):
        for y in range(0, w):
            pred = test.get_pixel(args.input, float(x), float(y), 0, h)
            res.append(pred)

    res = np.array(res)
    res = tensor(res.reshape((h, w)))

    transforms.ToPILImage()(res).save(args.output)
