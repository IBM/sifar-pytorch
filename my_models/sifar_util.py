import torch
from torch import nn
from einops import rearrange

from timm.models.layers import to_2tuple

def create_super_img(x, img_size, super_img_rows):
    input_size = x.shape[-2:]

    if not isinstance(img_size, tuple):
        img_size = to_2tuple(img_size)

    if input_size != img_size:
        x = nn.functional.interpolate(x, size=img_size, mode='bilinear')
    x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=super_img_rows, c=3)
    return x

def frames_to_super_image(x, super_img_rows, super_img_cols, img_h, img_w):
    x = rearrange(x, '(b th tw) (h w) c -> b (th h tw w) c', th=super_img_rows, tw=super_img_cols, h=img_h, w=img_w)
    return x

def super_image_to_frames(x, super_img_rows, super_img_cols, img_h, img_w):
    x = rearrange(x, 'b (th h tw w) c -> (b th tw) (h w) c', th=super_img_rows, tw=super_img_cols, h=img_h, w=img_w)
    return x

def pad_frames(x, duration, frame_padding):
    frame_num = duration - frame_padding
    x = x.view((-1, 3 * frame_num) + x.size()[2:])
    x_padding = torch.zeros((x.shape[0], 3 * frame_padding) + x.size()[2:]).cuda()
    x = torch.cat((x, x_padding), dim=1)
    assert x.shape[1] == 3 * duration, 'frame number %d not the same as adjusted input size %d' % (
    x.shape[1], 3 * duration)

    return x

def get_super_img_layout(duration, img_rows):
    return (img_rows, duration // img_rows)
