# Structure-Preserving Motion Estimation for Learned Video Compression

This is the official implementation and appendix of the paper:

Structure-Preserving Motion Estimation for Learned Video Compression. Han Gao, Jinzhong Cui, Mao Ye, Shuai Li, Yu Zhao, Xiatian Zhu. ACM Multimedia 2022. [[pdf](https://doi.org/10.1145/3503161.3548156)]

## TODO

* Upload appendix.pdf (Done);
* Upload codes (Done);
* Upload pretrained models (Done);
* Update README.md (Continuous maintenance).

## Overview

![Overview](https://github.com/gaohan-12/SPME/blob/main/Overview.png)

## Requirements

* Python==3.8
* Pytorch==1.9

## Data Preparation

### Testing dataset

* Download [HEVC dataset](), [UVG dataset](http://ultravideo.fi/#testsequences)(1080p/8bit/YUV/RAW) and [MCL-JCV dataset](http://mcl.usc.edu/mcl-jcv-dataset/), and convert them from YUV format to PNG format.

## Test

* Change the configs in class named `HEVC_dataset` of the file [dataset.py](https://github.com/gaohan-12/SPME/blob/main/dataset.py) to the path of the data to be tested, e.g. :

  ```
  root="/xxx/HEVC_dataset/Class_B", filelist="./Tools/filelists/B.txt"
  ```

* Run [test.py](https://github.com/gaohan-12/SPME/blob/main/test.py) for testing, in which the config named `--model_path` is the pretrained model path, and `--lambda_weight` is the lambda value of the prerained model, e.g. :

  ```
  python -u test.py --model_path="./Checkpoints/2048.pth" --lambda_weight=2048
  ```

## Acknowledgement

During implementation, we drawed on the experience of [CompressAI](https://github.com/InterDigitalInc/CompressAI), [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression) and [DCVC](https://github.com/DeepMC-DCVC/DCVC). The model weights of intra coding are from [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## Citation

If you find this paper useful, kindly cite:

```
@inproceedings{gao2022structure,
  title={Structure-Preserving Motion Estimation for Learned Video Compression},
  author={Gao, Han and Cui, Jinzhong and Ye, Mao and Li, Shuai and Zhao, Yu and Zhu, Xiatian},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia (MM’22)},
  year={2022}
}
```

## Contact

If any questions, kindly contact with Han Gao via e-mail: han.gao@std.uestc.edu.cn / gaohan_vc@163.com.
