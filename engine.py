# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from einops import rearrange
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma, reduce_tensor

import utils
from losses import DeepMutualLoss, ONELoss, SelfDistillationLoss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    simclr_criterion=None, simclr_w=0.,
                    branch_div_criterion=None, branch_div_w=0.,
                    simsiam_criterion=None, simsiam_w=0.,
                    moco_criterion=None, moco_w=0.,
                    byol_criterion=None, byol_w=0.,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False
                    ):
    # TODO fix this for finetuning
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    #criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        batch_size = targets.size(0)
        if simclr_criterion is not None or simsiam_criterion is not None or moco_criterion is not None or byol_criterion is not None:
            samples = [samples[0].to(device, non_blocking=True), samples[1].to(device, non_blocking=True)]
            targets = targets.to(device, non_blocking=True)
            ori_samples = [x.clone() for x in samples]  # copy the original samples

            if mixup_fn is not None:
                samples[0], targets_ = mixup_fn(samples[0], targets)
                if contrastive_nomixup:  # remain one copy for ce loss
                    samples[1] = ori_samples[0]
                    samples.append(ori_samples[1])
                elif hard_contrastive:
                    samples[1] = samples[1]
                else:
                    samples[1], _ = mixup_fn(samples[1], targets)
                targets = targets_

        else:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                # batch size has to be an even number
                if batch_size == 1:
                    continue
                if batch_size % 2 != 0:
                     samples, targets = samples[:-1], targets[:-1]
                samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples)
            if simclr_criterion is not None:
                # outputs 0: ce logits, bs x class, outputs 1: normalized embeddings of two views, bs x 2 x dim
                loss_ce = criterion(outputs[0], targets)
                loss_simclr = simclr_criterion(outputs[1])
                loss = loss_ce * (1.0 - simclr_w) + loss_simclr * simclr_w
            elif simsiam_criterion is not None:
                # outputs 0: ce logits, bs x class, outputs 1: normalized embeddings of two views, 4[bs x dim], [p1, z1, p2, z2]
                loss_ce = criterion(outputs[0], targets)
                loss_simsiam = simsiam_criterion(*outputs[1])
                loss = loss_ce * (1.0 - simsiam_w) + loss_simsiam * simsiam_w
            elif branch_div_criterion is not None:
                # outputs 0: ce logits, bs x class, outputs 1: embeddings of K branches, K[bs x dim]
                loss_ce = criterion(outputs[0], targets)
                loss_div = 0.0
                for i in range(0, len(outputs[1]), 2):
                    loss_div += torch.mean(branch_div_criterion(outputs[1][i], outputs[1][i + 1]))
                loss = loss_ce * (1.0 - branch_div_w) + loss_div * branch_div_w
            elif moco_criterion is not None:
                loss_ce = criterion(outputs[0], targets)
                loss_moco = moco_criterion(outputs[1][0], outputs[1][1])
                loss = loss_ce * (1.0 - moco_w) + loss_moco * moco_w
            elif byol_criterion is not None:
                loss_ce = criterion(outputs[0], targets)
                loss_byol = byol_criterion(*outputs[1])
                loss = loss_ce * (1.0 - byol_w) + loss_byol * byol_w
            else:
                if isinstance(criterion, (DeepMutualLoss, ONELoss, SelfDistillationLoss)):
                    loss, loss_ce, loss_kd = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if simclr_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_simclr=loss_simclr.item())
        elif simsiam_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_simsiam=loss_simsiam.item())
        elif branch_div_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_div=loss_div.item())
        elif moco_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_moco=loss_moco.item())
        elif byol_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_byol=loss_byol.item())
        elif isinstance(criterion, (DeepMutualLoss, ONELoss)):
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_kd=loss_kd.item())
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        batch_size = images.shape[0]
        #images = images.view((batch_size * num_crops * num_clips, -1) + images.size()[2:])
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)
            #loss = criterion(output, target)
        output = output.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
        else:
            outputs.append(output)
            targets.append(target)

        batch_size = images.shape[0]
        #metric_logger.update(loss=reduced_loss.item())
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    num_data = len(data_loader.dataset)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    import os
    if os.environ.get('TEST', False):
        import numpy as np
        print("dumping results...")
        tmp = outputs[:num_data].cpu().numpy()
        tt = targets[:num_data].cpu().numpy()
        np.savez("con_mix.npz", pred=tmp, gt=tt)
        
    real_acc1, real_acc5 = accuracy(outputs[:num_data], targets[:num_data], topk=(1, 5))
    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(real_acc1.item())
    metric_logger.meters['acc5'].update(real_acc5.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    #output = torch.cat(tensors_gather, dim=0)
    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')

    return output
