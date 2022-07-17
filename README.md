# Structure-Preserving Motion Estimation for Learned Video Compression

This is the official implementation and appendix of the paper:

Structure-Preserving Motion Estimation for Learned Video Compression. Han Gao, Jinzhong Cui, Mao Ye, Shuai Li, Yu Zhao, Xiatian Zhu. ACM Multimedia 2022. [Link(TODO)]

## TODO
* Upload appendix.pdf;
* Upload codes;
* Upload pretrained models;
* Update README.md.

## Overview
![Overview](https://github.com/gaohan-12/SPME/blob/main/Overview.png)

## Requirements
* Python==3.8
* Pytorch==1.9

## Data Preparation

### Testing dataset
* Download [HEVC dataset](), [UVG dataset](http://ultravideo.fi/#testsequences)(1080p/8bit/YUV/RAW) and [MCL-JCV dataset](http://mcl.usc.edu/mcl-jcv-dataset/), and convert them from YUV format to png format.

## Run

### Change the configs in the class *HEVC_dataset* of the file [dataset.py]() to the dataset path to be tested, e.g:
`root="/home/zhaoyu/HEVC_dataset/Class_B", filelist="./Tools/filelists/B.txt"`

## Acknowledgement
During implementation, we drawed on the experience of [CompressAI](https://github.com/InterDigitalInc/CompressAI), [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression) and [DCVC](https://github.com/DeepMC-DCVC/DCVC). The model weights of intra coding are from [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## Citation
If you find this paper useful, kindly cite:

## Contact
If any questions, kindly contact with Han Gao via e-mail: han.gao@std.uestc.edu.cn.
