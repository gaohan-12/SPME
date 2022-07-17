import torch
from torch import nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class Adp(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, aux_channel):
        super(Adp, self).__init__()
        self.DAB = DAB(n_feat, kernel_size, reduction, aux_channel)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_feat * 2, out_channels=n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, mv, mv_aux2, mv_aux3):
        """
        Using mv_aux2 and mv_aux3 to adjust or strength mv.

        :param mv: main motion vector: (B * C * H * W)
        :param mv_aux2: aux motion vector: (B * C * H * W)
        :param mv_aux3: another aux motion vector: (B * C * H * W)
        :return: adjusted motion vector: (B * C * H * W)
        """
        mv_aux = self.convs(torch.cat([mv_aux2, mv_aux3], 1))
        result = self.DAB(mv, mv_aux)

        return result


class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, aux_channel):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction, aux_channel)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction, aux_channel)
        self.conv1 = default_conv(n_feat, n_feat, kernel_size)
        self.conv2 = default_conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, y):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x, y))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2(out, y))
        out = self.conv2(out) + x

        return out


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction, aux_channel):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

        self.E = Encoder(aux_channel)

    def forward(self, x, y):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x.size()
        y = self.E(y)

        # branch 1
        # print(y.shape)
        kernel = self.kernel(y)
        # print(kernel.shape)
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        # print(kernel.shape)
        out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # print(out.shape)
        out = self.conv(out.view(b, -1, h, w))
        # print(out.shape)

        # branch 2
        out = out + self.ca(x, y)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(256, 256//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256 // reduction, 64, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(y[:, :, None, None])

        return x * att


class Encoder(nn.Module):
    def __init__(self, aux_channel):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(aux_channel, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return out