# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import warnings

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
import models
import my_models
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import simclr
import utils
from losses import DeepMutualLoss, ONELoss, MulMixturelLoss, SelfDistillationLoss
from vtab import DATASET_REGISTRY



from collections import OrderedDict

#from timm.models.vision_transformer import Block, Attention
from my_models import action_vit_ts, action_vit_hub, action_vit_swin


from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
from timm.models.resnet import Bottleneck, ResNet, default_cfgs
from video_dataset_config import get_dataset_config, DATASET_CONFIG

from main import get_args_parser

def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)


@register_model
def ecaresnet152d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet101d', pretrained, **model_args)


warnings.filterwarnings("ignore", category=UserWarning)
#torch.multiprocessing.set_start_method('spawn', force=True)

def summary(model, input_tensor, attention_cls):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict({'input_shape': 'N/A', 'output_shape': 'N/A', 'flops': 0, 'nb_params': 0})
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, attention_cls)):
                return
            if isinstance(input[0], (list, tuple)):
                return
            
            summary[m_key]['input_shape'] = list(input[0].size())
            batch_size = summary[m_key]['input_shape'][0]
            # summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]['nb_params'] = params


            if hasattr(module, 'kernel_size') and hasattr(module, 'out_channels') and hasattr(module, 'in_channels'):
                output_size = torch.prod(torch.LongTensor(summary[m_key]['output_shape'][1:]))
                flops_per_point = np.prod(module.kernel_size) * module.in_channels / module.groups
                summary[m_key]['flops'] = int(output_size * flops_per_point)
            else:
                if isinstance(module, nn.Linear):
                    if len(summary[m_key]['output_shape']) == 4:
                        summary[m_key]['flops'] = summary[m_key]['input_shape'][-1] * summary[m_key]['output_shape'][1] * summary[m_key]['output_shape'][2] * summary[m_key]['output_shape'][3]
                    elif len(summary[m_key]['output_shape']) == 3:
                        summary[m_key]['flops'] = summary[m_key]['input_shape'][-1] * summary[m_key]['output_shape'][1] * summary[m_key]['output_shape'][2]
                    elif len(summary[m_key]['output_shape']) == 2:
                        summary[m_key]['flops'] = summary[m_key]['input_shape'][-1] * summary[m_key]['output_shape'][-1]
                    else:
                        summary[m_key]['flops'] = 0
                elif isinstance(module, (attention_cls)):
                    n = summary[m_key]['input_shape'][1]
                    c = summary[m_key]['input_shape'][-1]
                    summary[m_key]['flops'] = (2 * (n * n * c))
                else:
                    summary[m_key]['flops'] = 0
            summary[m_key]['flops'] *= batch_size

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(input_tensor)
    # remove these hooks
    for h in hooks:
        h.remove()

    ret = ""
    ret += '-----------------------------------------------------------------------------------\n'
    line_new = '{:>24}  {:>25} {:>15} {:>15}\n'.format('Layer (type)', 'Output Shape', 'Param #', 'FLOPs #')
    ret += line_new
    ret += '===================================================================================\n'
    total_params = 0
    trainable_params = 0
    total_flops = 0
    for layer in summary:

        # if summary[layer]['flops'] == 0:
        #     continue
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>24}  {:>25} {:>15} {:>15}\n'.format(layer, str(summary[layer]['output_shape']), '{0:,}'.format(summary[layer]['nb_params']), '{0:,}'.format(summary[layer]['flops']))
        total_params += summary[layer]['nb_params']
        total_flops += summary[layer]['flops']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        ret += line_new

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ret += '===================================================================================\n'
    ret += 'Total flops: {0:,}\n'.format(total_flops)
    ret += 'Total params: {0:,}\n'.format(total_params)
    ret += 'Trainable params: {0:,}\n'.format(trainable_params)
    ret += 'Non-trainable params: {0:,}\n'.format(total_params - trainable_params)
    ret += '-----------------------------------------------------------------------------------'
    return ret, total_flops, total_params
    # return summary



def main(args):
    #utils.init_distributed_mode(args)
    #print(args)
    # Patch
    if not hasattr(args, 'hard_contrastive'):
        args.hard_contrastive = False
    if not hasattr(args, 'selfdis_w'):
        args.selfdis_w = 0.0


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset, args.use_lmdb)

    args.num_classes = num_classes
    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    if 'action_vit_ts' in args.model:
        Attention = action_vit_ts.Attention
    elif 'action_vit_hub' in args.model:
        Attention = action_vit_hub.Attention
    elif 'action_vit_swin' in args.model:
        Attention = action_vit_swin.WindowAttention
    else:
        from timm.models.vision_transformer import Block, Attention

    #print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        duration=args.duration,
        frame_cls_tokens=args.frame_cls_tokens,
        temporal_module_name=args.temporal_module_name,
        temporal_attention_only=args.temporal_attention_only,
        temporal_heads_scale=args.temporal_heads_scale,
        temporal_mlp_scale = args.temporal_mlp_scale,
        hpe_to_token = args.hpe_to_token,
        spatial_hub_size = args.spatial_hub_size,
        hub_attention=args.hub_attention,
        hub_aggregation=args.hub_aggregation,
        temporal_pooling = args.temporal_pooling,
        bottleneck = args.bottleneck,
        rel_pos = args.rel_pos,
        window_size=args.window_size,
        super_img_rows = args.super_img_rows,
        token_mask=not args.no_token_mask,
        online_learning = args.one_w >0.0 or args.dml_w >0.0,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )

    optimizer = create_optimizer(args, model)

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data = torch.randn((args.batch_size, 3 * args.duration, args.input_size, args.input_size), device=device, dtype=torch.float)
    data_ = torch.randn((1, 3 * args.duration, args.input_size, args.input_size), device=device, dtype=torch.float)
    with torch.no_grad():
        o, flops, params = summary(model, data_, Attention)
    #print(o)
    print(f"FLOPs: {flops}, Params: {params}")
    exit(0)
    #flops /= args.batch_size 
    target = torch.ones((args.batch_size), device=device, dtype=torch.long)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # training
    #print("Start!")
    if args.eval:
        model.eval()
        with torch.no_grad():
            for i in range(10):
                model(data)
        start.record()
        with torch.no_grad():
            for i in range(args.iters):
                model(data)
        end.record()
    else:
        for i in range(10):
            optimizer.zero_grad()
            out = model(data)
            loss = torch.mean(out)
            loss.backward()
            optimizer.step()
        start.record()
        for i in range(args.iters):
            optimizer.zero_grad()
            out = model(data)
            loss = torch.mean(out)
            loss.backward()
            optimizer.step()
        end.record()
    torch.cuda.synchronize()

    all_accs = {}
    try:
        log_paths = sorted(glob.glob(f'checkpoint/**/log.txt', recursive=True))
        for log_path in log_paths:
            model_name = os.path.basename(os.path.dirname(log_path))
            if args.model not in model_name:
                continue
            finish = False
            best_acc, best_epoch = 0, 0
            with open(log_path) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    stat = json.loads(line)
                    curr_acc = stat['test_acc1']
                    curr_epoch = stat['epoch']
                    if curr_acc > best_acc:
                        best_acc = curr_acc
                        best_epoch = curr_epoch
                    if curr_epoch == 299:
                        finish = True
                if not finish:
                    model_name = model_name + f"({curr_epoch})"
            all_accs[model_name] = best_acc 
    except Exception as e:
        print(e)
        model_name = "X_" + args.model
        best_acc = 0.0

    if all_accs == {}:
        all_accs[args.model] = 0
    for model_name, best_acc in all_accs.items():
        print(f"{model_name}\t{params / 1e6:.1f}\t{flops / 1e9:.1f}\t{best_acc:.2f}")
        print(f"{args.model}{'@Val' if args.eval else '@Train'}: {flops / 1e9:.1f} & {args.iters * args.batch_size / start.elapsed_time(end) * 1000.0:.1f} & {params / 1e6:.1f}")
    #print(f"{args.model}{'@Val' if args.eval else '@Train'}: {args.iters * args.batch_size / start.elapsed_time(end) * 1000.0:.2f} Images/second. FLOPs:{flops},Parameters: {params}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
