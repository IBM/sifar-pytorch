# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import logging
from einops import rearrange, reduce, repeat
from timm.models import resnet50, tv_resnet101, tv_resnet152
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.models as models

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    # ResNet
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth',
        interpolation='bicubic'),
    'resnet101': _cfg(url='', interpolation='bicubic'),
    'resnet101d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152': _cfg(url='', interpolation='bicubic'),
    'resnet200': _cfg(url='', interpolation='bicubic'),
}

class ConvActionModule(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, backbone=None, duration=8, img_size=224, in_chans=3, num_classes=1000, num_features=0,
                 super_img_rows=1, default_cfg=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.num_features = int(num_features)
        self.duration = duration
        self.num_classes = num_classes
        self.super_img_rows = super_img_rows
        self.default_cfg = default_cfg

        self.img_size = img_size
        self.frame_padding = self.duration % super_img_rows
        if self.frame_padding != 0:
            self.frame_padding = self.super_img_rows - self.frame_padding
            self.duration += self.frame_padding
#        assert (self.duration % super_img_rows) == 0, 'number of fames must be a multiple of the rows of the super image'

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        print('image_size:', self.img_size, 'padding frame:', self.frame_padding, 'super_img_size:', (super_img_rows, self.duration // super_img_rows))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def pad_frames(self, x):
        frame_num = self.duration - self.frame_padding
        x = x.view((-1,3*frame_num)+x.size()[2:])
        x_padding = torch.zeros((x.shape[0], 3*self.frame_padding) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.duration, 'frame number %d not the same as adjusted input size %d' % (x.shape[1], 3 * self.duration)

        return x

    def create_super_img(self, x):
        input_size = x.shape[-2:]
        if input_size != self.img_size:
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')

        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3)
        return x

    def forward_features(self, x):
        #        x = rearrange(x, 'b (t c) h w -> b c h (t w)', t=self.duration)
        # in evaluation, it's Bx(num_crops*num_cips*num_frames*3)xHxW
        if self.frame_padding > 0:
            x = self.pad_frames(x)
        else:
            x = x.view((-1,3*self.duration)+x.size()[2:])

        x = self.create_super_img(x)

        x = self.backbone.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def action_conv_resnet50(pretrained=False, **kwargs):

    num_features = 2048
    model = ConvActionModule(backbone=None, num_features=num_features, **kwargs)
    
    backbone = resnet50(pretrained=pretrained)
    model.backbone = backbone
    model.default_cfg = backbone.default_cfg

    return model

@register_model
def action_conv_resnet101(pretrained=False, **kwargs):

    num_features = 2048
    model = ConvActionModule(backbone=None, num_features=num_features, **kwargs)
    
    backbone = tv_resnet101(pretrained=pretrained)
    model.backbone = backbone
    model.default_cfg = backbone.default_cfg

    return model

@register_model
def action_conv_resnet152(pretrained=False, **kwargs):

    num_features = 2048
    model = ConvActionModule(backbone=None, num_features=num_features, **kwargs)
    
    backbone = tv_resnet152(pretrained=pretrained)
    model.backbone = backbone
    model.default_cfg = backbone.default_cfg

    return model

'''
@register_model
def action_tf_efficientnetv2_m_in21k(pretrained=False, **kwargs):
    num_features = 2048
    model_kwargs = dict(num_features=num_features, **kwargs)
    model = ConvActionModule(backbone=None, **model_kwargs)

    backbone = action_tf_efficientnetv2_m_in21k(pretrained=pretrained)
    print (backone)
    model.backbone = backbone
    model.default_cfg = backbone.default_cfga

    return model
'''
