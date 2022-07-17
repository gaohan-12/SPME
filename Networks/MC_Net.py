import torch
import torch.nn as nn
from Networks.Adaptive_module import *
from torchvision.ops.deform_conv import DeformConv2d


class MC_Net(nn.Module):
    def __init__(self):
        super(MC_Net, self).__init__()
        self.group = 8
        self.kernel_size = (3, 3)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
                      kernel_size=(3, 3), stride=1, padding=1)
            # nn.ReLU(inplace=True)
        )
        self.Deform_Conv2d = DeformConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.adp = DAB(n_feat=64, kernel_size=3, reduction=2, aux_channel=128)

    def forward(self, mv: torch.Tensor, ref_feature: torch.Tensor):
        offsets = self.head(mv)
        def_features = self.relu(self.Deform_Conv2d(ref_feature, offsets))

        output = self.adp(def_features, self.convs1(ref_feature))

        return output