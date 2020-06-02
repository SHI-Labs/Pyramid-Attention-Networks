# Pyramid Attention Networks for Image Restoration
This repository is for PANet and PA-EDSR introduced in the following paper

[Yiqun Mei](http://yiqunm2.web.illinois.edu/), [Yuchen Fan](https://scholar.google.com/citations?user=BlfdYL0AAAAJ&hl=en), [Yulun Zhang](http://yulunzhang.com/), [Jiahui Yu](https://jiahuiyu.com/), [Yuqian Zhou](https://yzhouas.github.io/), [Ding Liu](https://scholar.google.com/citations?user=PGtHUI0AAAAJ&hl=en), [Yun Fu](http://www1.ece.neu.edu/~yunfu/), [Thomas S. Huang](http://ifp-uiuc.github.io/) and [Humphrey Shi](https://www.humphreyshi.com/) "Pyramid Attention Networks for Image Restoration", [[Arxiv]](https://arxiv.org/abs/2004.13824) 

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) & [RNAN](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_1.1) with Titan X/1080Ti/V100 GPUs.

## Contents
1. [Introduction](#Introduction)
2. [Tasks](#Tasks)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)


## Introduction
Self-similarity refers to the image prior widely used in image restoration algorithms that small but similar patterns tend to occur at different locations and scales. However, recent advanced deep convolutional neural network based methods for image restoration do not take full advantage of self-similarities by relying on self-attention neural modules that only process information at the same scale. To solve this problem, we present a novel Pyramid Attention module for image restoration, which captures long-range feature correspondences from a multi-scale feature pyramid. Inspired by the fact that corruptions, such as noise or compression artifacts, drop drastically at coarser image scales, our attention module is designed to be able to borrow clean signals from their "clean" correspondences at the coarser levels. The proposed pyramid attention module is a generic building block that can be flexibly integrated into various neural architectures. Its effectiveness is validated through extensive experiments on multiple image restoration tasks: image denoising, demosaicing, compression artifact reduction, and super resolution. Without any bells and whistles, our PANet (pyramid attention module with simple network backbones) can produce state-of-the-art results with superior accuracy and visual quality.
![block](/Figs/block.png)

## Tasks
### Color Image Denoising 
![PSNR_DN_RGB](/Figs/PSNR_DN_RGB.png)
![Visual_DN_RGB](/Figs/Visual_DN_RGB.png)

More details at [DN_RGB](https://github.com/SHI-Labs/Pyramid-Attention-Networks/tree/master/DN_RGB).
### Image Demosaicing 
![PSNR_Demosaic](/Figs/PSNR_Demosaic.png)
![Visual_Demosaic](/Figs/Visual_Demosaic.png)

More details at [Demosaic](https://github.com/SHI-Labs/Pyramid-Attention-Networks/tree/master/Demosaic).
### Image Compression Artifact Reduction 
![PSNR_CAR](/Figs/PSNR_CAR.png)
![Visual_CAR](/Figs/Visual_CAR.png)

More details at [CAR](https://github.com/SHI-Labs/Pyramid-Attention-Networks/tree/master/CAR).
### Image Super-resolution 
![PSNR_SR](/Figs/PSNR_SR.png)
![Visual_SR](/Figs/Visual_SR.png)

More details at [SR](https://github.com/SHI-Labs/Pyramid-Attention-Networks/tree/master/SR).

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
