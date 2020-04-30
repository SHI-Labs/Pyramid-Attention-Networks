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

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) All the pretrained models and visual results can be downloaded from [Google Drive](https://drive.google.com/open?id=1q9iUzqYX0fVRzDu4J6fvSPRosgOZoJJE).

2. Cd to 'PANet-PyTorch/[Task]/code', run the following scripts to train models.

    **You can use scripts in file 'demo.sb' to train and test models for our paper.**

    ```bash
    # Example Usage: 
    python main.py --n_GPUs 4 --rgb_range 1 --reset --save_models --lr 1e-4 --decay 200-400-600-800 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model PAEDSR --scale 2 --patch_size 96 --save EDSR_PA_x2 --data_train DIV2K
    ```
## Test
### Quick start

1. Cd to 'PANet-PyTorch/[Task]/code', run the following scripts.

    **You can use scripts in file 'demo.sb' to produce results for our paper.**

    ```bash
    # No self-ensemble, use different testsets to reproduce the results in the paper.
    # Example Usage: 
    python main.py --model PAEDSR --data_test Set5+Set14+B100+Urban100 --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1 --pre_train ../model_x2.pt --test_only --chop 

    ```

### The whole test pipeline
1. Prepare benchmark datasets [from SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) 

2. Conduct image SR. 

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
