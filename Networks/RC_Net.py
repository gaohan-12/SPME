import torch
import torch.nn as nn
from Common.Utils2048 import layer_similarity_map


class RC_Net(nn.Module):
    def __init__(self):
        super(RC_Net, self).__init__()

        self.res_blocks = nn.Sequential(
            ResBlock(channels=64),
            ResBlock(channels=64),
            ResBlock(channels=64)
        )
        self.bottle = nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                         kernel_size=(5, 5), stride=2, padding=2, output_padding=1)

    def forward(self, features: torch.Tensor):

        res = self.res_blocks(features)
        output = self.bottle(features + res)
        return output


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        y = x + self.block(x)
        return self.relu(y)
