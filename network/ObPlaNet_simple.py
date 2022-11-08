"""该模型对应简单利用动态特征的方法, 将最后两次前景特征经过conv与背景特征融合"""
from sqlalchemy import true
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

import sys

sys.path.append("..")
from network.BaseBlocks import BasicConv2d
from network.tensor_ops import cus_sample, upsample_add

from network.MyModules import (
    DDPM,
    DenseTransLayer,
)

from network.OwnModules import simpleDFN

from backbone.ResNet import Backbone_ResNet50_in1, Backbone_ResNet50_in3, Backbone_ResNet18_in1, Backbone_ResNet18_in3, \
    Backbone_ResNet18_in3_1
from backbone.VGG import (
    Backbone_VGG19_in1,
    Backbone_VGG19_in3,
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)


class ObPlaNet_resnet18(nn.Module):
    def __init__(self, pretrained=True, ks=3, scale=3):
        super(ObPlaNet_resnet18, self).__init__()
        self.Eiters = 0
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.to_pil = transforms.ToPILImage()
        self.scale = scale

        self.add_mask = True

        (
            self.bg_encoder1,
            self.bg_encoder2,
            self.bg_encoder4,
            self.bg_encoder8,
            self.bg_encoder16,
        ) = Backbone_ResNet18_in3(pretrained=pretrained)
        
        # Lock background encooder
        for p in self.parameters():
            p.requires_grad = False

        (
            self.fg_encoder1,
            self.fg_encoder2,
            self.fg_encoder4,
            self.fg_encoder8,
            self.fg_encoder16,
            self.fg_encoder32,
        ) = Backbone_ResNet18_in3_1(pretrained=pretrained)

        if self.add_mask:
            self.mask_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # dynamic conv
        self.fg_trans16 = nn.Conv2d(512, 64, 1)
        self.fg_trans8 = nn.Conv2d(256, 64, 1)
        self.selfdc_16 = simpleDFN(64, 64, 512, ks, 4)
        self.selfdc_8 = simpleDFN(64, 64, 512, ks, 4)

        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(512, 2, 1)
        print("dynamic conv")

    def forward(self, bg_in_data, fg_in_data, mask_in_data=None, mode='val'):
        """
        Args:
            bg_in_data: (batch_size * 3 * H * W) 背景图片特征
            fg_in_data: (batch_size * 3 * H * W) 前景物体特征
            mask_in_data: (batch_size * 1 * H * W) 前景mask特征
            mode: 当前模式 train / val, 前者为训练, 后者为测试
        """
        if ('train' == mode):
            self.Eiters += 1
        # Unet 前半部分,背景和前景特征提取
        black_mask = torch.zeros(mask_in_data.size()).to(mask_in_data.device)
        bg_in_data_ = torch.cat([bg_in_data, black_mask], dim=1)
        bg_in_data_1 = self.bg_encoder1(bg_in_data_)  # torch.Size([2, 64, 128, 128])
        #del bg_in_data
        fg_cat_mask = torch.cat([fg_in_data, mask_in_data], dim=1)
        fg_in_data_1 = self.fg_encoder1(fg_cat_mask)  # torch.Size([2, 64, 128, 128])
        #del fg_in_data

        bg_in_data_2 = self.bg_encoder2(bg_in_data_1)  # torch.Size([2, 64, 64, 64])
        fg_in_data_2 = self.fg_encoder2(fg_in_data_1)  # torch.Size([2, 64, 128, 128])
        bg_in_data_4 = self.bg_encoder4(bg_in_data_2)  # torch.Size([2, 128, 32, 32])
        fg_in_data_4 = self.fg_encoder4(fg_in_data_2)  # torch.Size([2, 64, 64, 64])
        del fg_in_data_1, fg_in_data_2

        bg_in_data_8 = self.bg_encoder8(bg_in_data_4)  # torch.Size([2, 256, 16, 16])
        fg_in_data_8 = self.fg_encoder8(fg_in_data_4)  # torch.Size([2, 128, 32, 32])
        bg_in_data_16 = self.bg_encoder16(bg_in_data_8)  # torch.Size([2, 512, 8, 8])
        fg_in_data_16 = self.fg_encoder16(fg_in_data_8)  # torch.Size([2, 256, 16, 16])
        fg_in_data_32 = self.fg_encoder32(fg_in_data_16)  # torch.Size([2, 512, 8, 8])

        in_data_8_aux = self.fg_trans8(fg_in_data_16)  # torch.Size([2, 64, 16, 16])
        in_data_16_aux = self.fg_trans16(fg_in_data_32)  # torch.Size([2, 64, 8, 8])

        # Unet 后半部分
        bg_out_data_16 = bg_in_data_16  # torch.Size([2, 512, 8, 8])
        # 先降维再上采样（上采样的过程中输入与输出相加）
        bg_out_data_8 = self.upsample_add(self.upconv16(bg_out_data_16), bg_in_data_8)  # torch.Size([2, 256, 16, 16])
        bg_out_data_4 = self.upsample_add(self.upconv8(bg_out_data_8), bg_in_data_4)  # torch.Size([2, 128, 32, 32])
        bg_out_data_2 = self.upsample_add(self.upconv4(bg_out_data_4), bg_in_data_2)  # torch.Size([2, 64, 64, 64])
        bg_out_data_1 = self.upsample_add(self.upconv2(bg_out_data_2), bg_in_data_1)  # torch.Size([2, 64, 128, 128])
        del bg_out_data_2, bg_out_data_4, bg_out_data_8, bg_out_data_16

        bg_out_data = self.upconv1(self.upsample(bg_out_data_1, scale_factor=2))  # torch.Size([2, 64, 256, 256])

        # 前景和背景特征融合
        fuse_out = self.upsample_add(self.selfdc_16(bg_out_data_1, in_data_8_aux), \
                                     self.selfdc_8(bg_out_data, in_data_16_aux))  # torch.Size([2, 64, 256, 256])

        out_data = self.classifier(fuse_out)  # torch.Size([2, 2, 256, 256])

        return out_data, fuse_out

if __name__ == "__main__":
    a = torch.randn((2, 3, 256, 256))
    b = torch.randn((2, 3, 256, 256))
    c = torch.randn((2, 1, 256, 256))

    model = ObPlaNet_resnet18()
    x, y = model(a, b, c)
    print(x.size())
    print(y.size())

