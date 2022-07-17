import itertools
import math
import os
import numpy as np
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = " 2 "

import torch
import torch.nn as nn
import argparse
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from compressai.zoo import cheng2020_attn, cheng2020_anchor, bmshj2018_hyperprior

from dataset1 import HEVCDataSet, UVGDataSet, VTL_DataSet
from SPMENet import SPME_Net


gpu_num = torch.cuda.device_count()


def parse_args():
    parser = argparse.ArgumentParser(description="Testing configs")

    parser.add_argument("--lambda_weight", type=int, default=2048, help="the lambda value")
    parser.add_argument("--model_path", type=str, default="./Checkpoints/2048.pth", help="the pre-trained model path")

    args = parser.parse_args()
    return args


lambda_I_quality_map = {256: 3,
                        512: 4,
                        1024: 5,
                        2048: 6}


def cal_rd_cost(distortion: torch.Tensor, bpp: torch.Tensor, lambda_weight: float = 1024):
    rd_cost = lambda_weight * distortion + bpp
    return rd_cost


def cal_bpp(likelihood: torch.Tensor, num_pixels: int):
    bpp = torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
    return bpp


def cal_distoration(A: torch.Tensor, B:torch.Tensor):
    dis = nn.MSELoss()
    return dis(A, B)


def cal_psnr(distortion: torch.Tensor):
    psnr = -10 * torch.log10(distortion)
    return psnr


def Var(x):
    return Variable(x.cuda())


def test():
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)

    sumbpp = 0
    sumbpp_mv_y = 0
    sumbpp_mv_z = 0
    sumbpp_res_y = 0
    sumbpp_res_z = 0
    sumpsnr = 0
    eval_step = 0
    gop_num = 0
    avg_loss = torch.zeros(size=[1, ])
    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("testing : %d/%d" % (batch_idx, len(test_loader)))
        input_images = input[0]
        seqlen = input_images.size()[0]

        net.eval()
        with torch.no_grad():
            for i in range(seqlen):

                if i == 0:
                    I_frame = input_images[i, :, :, :].cuda()
                    print(I_frame.shape)
                    num_pixels = 1 * I_frame.shape[1] * I_frame.shape[2]
                    arr = I_codec(torch.unsqueeze(I_frame, 0))
                    I_rec = arr['x_hat']
                    I_likelihood_y = arr["likelihoods"]['y']
                    I_likelihood_z = arr["likelihoods"]['z']

                    ref_image = I_rec.clone().detach()
                    y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy()
                    z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                    psnr = cal_psnr(distortion=cal_distoration(I_rec, I_frame)).cpu().detach().numpy()
                    bpp = y_bpp + z_bpp

                    print("------------------ GOP {0} --------------------".format(batch_idx + 1))
                    print("I frame:  ", bpp, "\t", psnr)

                    gop_num += 1

                else:
                    cur_frame = input_images[i, :, :, :].cuda()
                    ref_left = input_images[i - 1, :, :, :].cuda()
                    cur_frame, ref_image, ref_left = Var(torch.unsqueeze(cur_frame, 0)), Var(
                        torch.unsqueeze(ref_image, 0)), Var(torch.unsqueeze(ref_left, 0))
                    mv_likelihood, mv_hyper_likelihood, pred_frame, res_likelihood, res_hyper_likelihood, recon_frame, res, res_hat = net(
                        cur_frame, ref_image, ref_left,
                        stage="REC")

                    ref_image = recon_frame
                    # calculate rd cost
                    mv_bpp = cal_bpp(likelihood=mv_likelihood, num_pixels=num_pixels).cpu().detach().numpy()
                    mv_hyper_bpp = cal_bpp(likelihood=mv_hyper_likelihood, num_pixels=num_pixels).cpu().detach().numpy()
                    res_bpp = cal_bpp(likelihood=res_likelihood, num_pixels=num_pixels).cpu().detach().numpy()
                    res_hyper_bpp = cal_bpp(likelihood=res_hyper_likelihood,
                                            num_pixels=num_pixels).cpu().detach().numpy()
                    distortion = torch.mean((cur_frame - recon_frame).pow(2))
                    rd_cost = cal_rd_cost(distortion=distortion,
                                          bpp=res_bpp + mv_bpp + mv_hyper_bpp + res_hyper_bpp).cpu()
                    psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                    bpp = mv_bpp + mv_hyper_bpp + res_hyper_bpp + res_bpp

                    print("P{0} frame: ".format(i), bpp, "\t", psnr)

                    sumbpp_mv_y += mv_bpp
                    sumbpp_mv_z += mv_hyper_bpp
                    sumbpp_res_y += res_bpp
                    sumbpp_res_z += res_hyper_bpp

                sumbpp += bpp
                sumpsnr += psnr
                eval_step += 1

    sumbpp /= eval_step
    sumpsnr /= eval_step
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)
    print('\nEpoch {0}  Average MSE={1}  Eval Step={2}\n'.format(str(0), str(avg_loss.data), int(eval_step)))
    log = "HEVC_Class_D  : average bpp : %.6lf, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, " \
          " res_y_bpp : %.6lf, res_z_bpp : %.6lf, average psnr : %.6lf\n" % (
        sumbpp, sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr)
    print(log)


if __name__ == "__main__":

    args = parse_args()
    I_codec = cheng2020_anchor(quality=lambda_I_quality_map[args.lambda_weight], metric='mse', pretrained=True).cuda()
    I_codec.eval()
    model = SPME_Net()
    pretrained_dict = torch.load(args.model_path)
    model_dict = model.state_dict()
    ckpt = pretrained_dict
    pretrained_net = {k: v for k, v in ckpt["net"].items() if k in model_dict}
    model_dict.update(pretrained_net)
    model.load_state_dict(model_dict, strict=False)

    net = model.cuda()

    print("Number of Total Parameters:", sum(x.numel() for x in net.parameters()))
    global test_dataset
    test_dataset = HEVCDataSet()
    test()
    exit(0)

