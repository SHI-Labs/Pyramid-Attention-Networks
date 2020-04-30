# Pyramid Attention for Image Restoration
This repository is for PANet and PA-EDSR introduced in the following paper

[Yiqun Mei](http://yiqunm2.web.illinois.edu/), [Yuchen Fan](https://scholar.google.com/citations?user=BlfdYL0AAAAJ&hl=en), [Yulun Zhang](http://yulunzhang.com/), [Jiahui Yu](https://jiahuiyu.com/), [Yuqian Zhou](https://yzhouas.github.io/), [Ding Liu](https://scholar.google.com/citations?user=PGtHUI0AAAAJ&hl=en), [Yun Fu](http://www1.ece.neu.edu/~yunfu/), [Thomas S. Huang](http://ifp-uiuc.github.io/) and [Honghui Shi](https://www.humphreyshi.com/) "Pyramid Attention for Image Restoration", [[Arxiv]](https://arxiv.org/abs/2004.13824) 

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) & [RNAN](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_1.1) with Titan X/1080Ti/V100 GPUs.

## Contents
1. [Train](#train)
2. [Test](#test)
3. [Results](#results)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path.

3. Organize training data like:
```bash
DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X10
│   └── X20
│   └── X30
│   └── X40
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X10
    └── X20
    └── X30
    └── X40
```
For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) All the pretrained models and visual results can be downloaded from [Google Drive](https://drive.google.com/open?id=1q9iUzqYX0fVRzDu4J6fvSPRosgOZoJJE).

2. Cd to 'PANet-PyTorch/[Task]/code', run the following scripts to train models.

    **You can use scripts in file 'demo.sb' to train and test models for our paper.**

    ```bash
    # Example Usage: Q=10 
    python main.py --n_GPUs 2 --batch_size 16 --lr 1e-4 --decay 200-400-600-800 ---save_models --n_resblocks 80 --model PANET --scale 10 --patch_size 48 --save PANET_Q10 --n_feats 64 --data_train DIV2K --chop
    ```
## Test
### Quick start

1. Cd to 'PANet-PyTorch/[Task]/code', run the following scripts.

    **You can use scripts in file 'demo.sb' to produce results for our paper.**

    ```bash
    # No self-ensemble, use different testsets (Classic5, LIVE1) to reproduce the results in the paper.
    # Example Usage: Q=40
    python main.py --model PANET --save_results --n_GPUs 1 --chop --data_test classic5+LIVE1 --scale 40 --n_resblocks 80 --n_feats 64 --pre_train ../Q40.pt --test_only

    ```

### The whole test pipeline
1. Prepare test data. Organize training data like:
```bash
benchmark/
├── testset1
│   └── HR 
│   └── LR_bicubic
│   	└── X10
│   	└── ..
├── testset2
```

2. Conduct image CAR. 

    See **Quick start**
3. Evaluate the results.

    Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{mei2020pyramid,
  title={Pyramid Attention Networks for Image Restoration},
  author={Mei, Yiqun and Fan, Yuchen and Zhang, Yulun and Yu, Jiahui and Zhou, Yuqian and Liu, Ding and Fu, Yun and Huang, Thomas S and Shi, Honghui},
  journal={arXiv preprint arXiv:2004.13824},
  year={2020}
}
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch), [RNAN](https://github.com/yulunzhang/RNAN) and [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch). We thank the authors for sharing their codes.
