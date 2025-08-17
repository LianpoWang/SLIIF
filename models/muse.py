import torch
import torch.nn as nn

import models
from models import register


# --------------------------MUSE_BLOCK------------------------------- #

class MUSE_BLOCK(nn.Module):
    def __init__(self, encoder_spec, channel):
        super(MUSE_BLOCK, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        self.conv_1_2 = nn.Conv2d(in_channels=channel * 3, out_channels=channel * 3, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        self.conv_1_3 = nn.Conv2d(in_channels=channel * 9, out_channels=channel * 9, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.conv_3_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=channel * 3, out_channels=channel * 3, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_3_3 = nn.Conv2d(in_channels=channel * 9, out_channels=channel * 9, kernel_size=3, stride=1, padding=1,
                                  bias=True)

        self.conv_5_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=channel * 3, out_channels=channel * 3, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_5_3 = nn.Conv2d(in_channels=channel * 9, out_channels=channel * 9, kernel_size=5, stride=1, padding=2,
                                  bias=True)

        self.confusion = nn.Conv2d(in_channels=channel * 27, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = models.make(encoder_spec)

    def forward(self, x):
        identity_data = x

        output_1_1 = self.relu(self.conv_1_1(x))
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_1 = torch.cat([output_1_1, output_3_1, output_5_1], 1)

        output_1_2 = self.relu(self.conv_1_2(input_1))
        output_3_2 = self.relu(self.conv_3_2(input_1))
        output_5_2 = self.relu(self.conv_5_2(input_1))

        input_2 = torch.cat([output_1_2, output_3_2, output_5_2], 1)

        output_1_3 = self.relu(self.conv_1_3(input_2))
        output_3_3 = self.relu(self.conv_3_3(input_2))
        output_5_3 = self.relu(self.conv_5_3(input_2))

        output = torch.cat([output_1_3, output_3_3, output_5_3], 1)

        output = self.confusion(output)

        output = self.encoder(output)

        output = torch.add(output, identity_data)
        return output


@register('muse')
class MUSE(nn.Module):
    def __init__(self, endoder):
        super(MUSE, self).__init__()
        self.channel = 64
        self.muse_block = MUSE_BLOCK(endoder, self.channel)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=3, stride=1, padding=1,
                                    bias=True)
        self.out_dim = self.channel

    def forward(self, x):
        out = self.conv_input(x)
        LR = out
        out = self.muse_block(out)
        out = self.muse_block(out)
        out = self.muse_block(out)

        out = torch.add(LR, out)

        return out
