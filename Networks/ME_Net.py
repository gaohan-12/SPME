import torch
import torch.nn as nn


class ME_Net(nn.Module):
    def __init__(self, kernel_size):
        super(ME_Net, self).__init__()

        self.feature_extraction_cur = Feature_extraction_module()
        self.feature_extraction_ref = Feature_extraction_module()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
        )

    def forward(self, cur_frame: torch.Tensor, ref_frame: torch.Tensor):
        f_cur = self.feature_extraction_cur(cur_frame)
        f_ref = self.feature_extraction_ref(ref_frame)
        concat = torch.cat([f_cur, f_ref], dim=1)
        offsets = self.convs(concat)
        return offsets, f_ref, f_cur


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


class Feature_extraction_module(nn.Module):
    def __init__(self):
        super(Feature_extraction_module, self).__init__()

        self.head = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            ResBlock(channels=64),
            ResBlock(channels=64),
            ResBlock(channels=64)
        )

    def forward(self, frame: torch.Tensor):
        conv1 = self.relu(self.head(frame))
        res = self.res_blocks(conv1)
        output = conv1 + res
        return output
