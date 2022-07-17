import torch
import torch.nn as nn
import os
from Networks.ME_Net import ME_Net, ResBlock
from Networks.MC_Net import *
from Networks.RC_Net import RC_Net
from Networks.FVC_NIPS_Codec import Codec_res, Codec_mv
from Networks.Adaptive_module import *


class SPME_Net(nn.Module):
    def __init__(self):
        super(SPME_Net, self).__init__()
        self.ME_Net = ME_Net(kernel_size=3)
        self.adap = DAB(n_feat=64, kernel_size=3, reduction=2, aux_channel=64)
        self.MC_Net = MC_Net()
        self.RC_Net = RC_Net()

        self.MV_Codec = Codec_mv(input_channel=64, output_channel=128)
        self.RES_Codec = Codec_mv(input_channel=64, output_channel=64)

    def forward(self, cur_frame: torch.Tensor, ref_list: torch.Tensor, ref_left, stage: str):
        ref_frame = ref_list[-1]

        if stage == "REC":

            mv, ref_feature, cur_feature = self.ME_Net(cur_frame, ref_frame)
            mv_aux, _, _ = self.ME_Net(cur_frame, ref_left)
            mv = self.adap(mv, mv_aux)

            mv_hat, mv_likelihood, mv_hyper_likelihood = self.MV_Codec(mv)

            pred_features = self.MC_Net(mv_hat, ref_feature)
            pred_frame = self.RC_Net(pred_features)
            pred_frame = torch.clamp(pred_frame, 0., 1.)

            res = cur_feature - pred_features
            res_hat, res_likelihood, res_hyper_likelihood = self.RES_Codec(res)

            rec_feature = res_hat + pred_features
            recon_frame = self.RC_Net(rec_feature)

            recon_frame = torch.clamp(recon_frame, 0., 1.)

            return mv_likelihood, mv_hyper_likelihood, pred_frame, res_likelihood, res_hyper_likelihood, recon_frame, res, res_hat

