import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from IPython import embed

"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
It is proposed in the ICCV2023 paper  
`S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

Arguments:
    s3im_kernel_size (int): kernel size in ssim's convolution(default: 4)
    s3im_stride (int): stride in ssim's convolution(default: 4)
    s3im_repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
    s3im_patch_height (height): height of virtual patch(default: 64)
"""

class S3IM(torch.nn.Module):
    def __init__(self, s3im_kernel_size = 4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=103, size_average = True):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)

    
    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor([exp(-(x - s3im_kernel_size//2)**2/float(2*sigma**2)) for x in range(s3im_kernel_size)])
        return gauss/gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
        return s3im_kernel

    def _ssim(self, img1, img2, s3im_kernel, s3im_kernel_size, channel, size_average = True, s3im_stride=None):
        mu1 = F.conv2d(img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)
        mu2 = F.conv2d(img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu2_sq
        sigma12 = F.conv2d(img1*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def ssim_loss(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)

            if img1.is_cuda:
                s3im_kernel = s3im_kernel.cuda(img1.get_device())
            s3im_kernel = s3im_kernel.type_as(img1)

            self.s3im_kernel = s3im_kernel
            self.channel = channel


        return self._ssim(img1, img2, s3im_kernel, self.s3im_kernel_size, channel, self.size_average, s3im_stride=self.s3im_stride)

    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss
    
# if __name__ == "__main__":
#     s3im = S3IM()
#     src = torch.ones(3, 512, 512).flatten(1).permute(0, 1)
#     embed()
#     tar = torch.ones(3, 512, 512).flatten(1).permute(0, 1)
#     loss = s3im(src, tar)
#     print(loss)
