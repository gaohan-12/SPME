import os
import torch.nn as nn
import imageio
import numpy as np
import glob
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from Tools.ms_ssim_torch import *
from torchvision.utils import save_image


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))


def cal_distoration(A: torch.Tensor, B:torch.Tensor):
    return nn.MSELoss(A, B)


def cal_psnr(distortion: torch.Tensor):
    psnr = -10 * torch.log10(distortion)
    return psnr


class HEVCDataSet(data.Dataset):
    def __init__(self, root="", filelist="./Tools/filelists/B.txt", testfull=True):
        with open(filelist) as f:
            folders = f.readlines()
        self.input = []
        self.hevcclass = []
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
                if cnt == 100:
                    break
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 10 + j + 1).zfill(3) + '.png'))
                self.input.append(inputpath)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0  # [3, h, w]
            h = int((input_image.shape[1] // 64) * 64)
            w = int((input_image.shape[2] // 64) * 64)
            input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images


class UVGDataSet(data.Dataset):
    def __init__(self, root="", filelist="./Tools/filelists/UVG.txt", testfull=True):
        with open(filelist) as f:
            folders = f.readlines()
        self.input = []
        self.hevcclass = []
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3) + '.png'))
                self.input.append(inputpath)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
            h = int((input_image.shape[1] // 64) * 64)
            w = int((input_image.shape[2] // 64) * 64)
            input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images


class VTL_DataSet(data.Dataset):
    def __init__(self, root="", testfull=True):
        self.input = []
        gop = 12
        folderlist = glob.glob(root + "/*")
        for folder in folderlist:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
                if cnt == 300:
                    break
            if testfull:
                framerange = cnt // gop
            else:
                framerange = 1
            for i in range(framerange):
                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(seq, 'im' + str(i * gop + j + 1).zfill(3) + '.png'))
                self.input.append(inputpath)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0  # [3, h, w]
            h = int((input_image.shape[1] // 64) * 64)
            w = int((input_image.shape[2] // 64) * 64)
            input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images