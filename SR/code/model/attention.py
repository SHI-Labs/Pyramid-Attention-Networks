import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from model import common
from utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding

class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv,channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv,channel, channel,1,bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        #theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base,1,dim=0)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            #sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            #sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            #group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, C, k, k]
            #normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)
            
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)/4.
            y.append(yi)
      
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        return y