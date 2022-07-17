import torch
import torch.nn as nn
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.priors import ScaleHyperprior
from compressai.models.priors import JointAutoregressiveHierarchicalPriors
import torch.nn.functional as F
from Networks.ME_Net import ResBlock
from Networks.zz import ConvGRU


class Enc_Unit(nn.Module):
    def __init__(self, input_channel: int = 64):
        self.input_channel = input_channel
        super(Enc_Unit, self).__init__()
        self.head = nn.Conv2d(in_channels=self.input_channel, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            ResBlock(channels=128),
            ResBlock(channels=128),
            ResBlock(channels=128)
        )

    def forward(self, inputs: torch.Tensor):
        conv1 = self.relu(self.head(inputs))
        res = self.res_blocks(conv1)
        output = conv1 + res
        return output


class Dec_Unit(nn.Module):
    def __init__(self):
        super(Dec_Unit, self).__init__()
        self.res_blocks = nn.Sequential(
            ResBlock(channels=128),
            ResBlock(channels=128),
            ResBlock(channels=128)
        )
        self.bottle = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                         kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor):
        ress = self.res_blocks(input) + input
        output = self.relu(self.bottle(ress))
        return output


class Dec_Unit_AME(Dec_Unit):
    def __init__(self):
        super().__init__()
        self.bottle = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                         kernel_size=(3, 3), stride=1, padding=1, output_padding=0)


class Codec_res(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, input_channel=3, output_channel=3, N=128, M=128, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            Enc_Unit(input_channel=input_channel),
            Enc_Unit(input_channel=128),
            Enc_Unit(input_channel=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        )

        self.g_s = nn.Sequential(
            Dec_Unit(),
            Dec_Unit(),
            Dec_Unit(),
            nn.ConvTranspose2d(in_channels=128, out_channels=output_channel,
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, F.relu(scales_hat) + 0.00000001, means=means_hat)
        x_hat = self.g_s(y_hat)

        return x_hat, y_likelihoods, z_likelihoods

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class Codec_mv(Codec_res):
    def __init__(self, input_channel=64, output_channel=128, N=128, M=128, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            Enc_Unit(input_channel=input_channel),
            Enc_Unit(input_channel=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        )

        self.g_s = nn.Sequential(
            Dec_Unit(),
            Dec_Unit(),
            nn.ConvTranspose2d(in_channels=128, out_channels=output_channel,
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        )


class Codec_condition(JointAutoregressiveHierarchicalPriors):

    def __init__(self, input_channel=128, output_channel=64, N=128, M=128, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            Enc_Unit(input_channel=input_channel),
            Enc_Unit(input_channel=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        )

        self.g_s = nn.Sequential(
            Dec_Unit(),
            Dec_Unit(),
            nn.ConvTranspose2d(in_channels=128, out_channels=output_channel,
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        # self.context_prediction = MaskedConv2d(
        #     M, 2 * M, kernel_size=5, padding=2, stride=1
        # )
        self.pre_prediction = nn.Sequential(
            Enc_Unit(input_channel=64),
            Enc_Unit(input_channel=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1)
        )

        self.post = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(channels=128),
            ResBlock(channels=128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, content):
        predict = content[1]
        x = torch.cat([content[0], content[1]], dim=1)
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        pre_params = self.pre_prediction(predict)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, pre_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, F.relu(scales_hat) + 0.00000001, means=means_hat)
        x_hat = self.g_s(y_hat)

        concat = torch.cat([x_hat, predict], dim=1)
        x_hat = self.post(concat)

        return x_hat, y_likelihoods, z_likelihoods


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )






