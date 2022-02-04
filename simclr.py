"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class SimSiamLoss(nn.Module):

    def __init__(self, version='simplified'):
        super().__init__()
        self.version = version

    def forward(self, z1, z2, p1, p2):

        def _loss(p, z, version):
            if version == 'original':
                z = z.detach()  # stop gradient
                p = F.normalize(p, dim=1)  # l2-normalize
                z = F.normalize(z, dim=1)  # l2-normalize
                return -(p * z).sum(dim=1).mean()
            elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
                return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
            else:
                raise Exception

        out = torch.mean(_loss(p1, z2, self.version) / 2 + _loss(p2, z1, self.version) / 2)
        return out


class BYOLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, online_pred_one, online_pred_two, target_proj_one, target_proj_two):

        def loss_fn(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


#  https://github.com/Spijkervet/SimCLR/blob/654f05f107ce17c0a9db385f298a2dc6f8b3b870/modules/nt_xent.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, temperature=0.07):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, zz):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        z_i, z_j = zz[:, 0], zz[:, 1]
        batch_size = z_i.shape[0]
        world_size = dist.get_world_size()
        N = 2 * batch_size * world_size

        mask = self.mask_correlated_samples(batch_size, world_size)

        z = torch.cat((z_i, z_j), dim=0)
        if world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size * world_size)

        sim_j_i = torch.diag(sim, -batch_size * world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N, device=positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


def all_gather(tensor, expand_dim=0, num_replicas=None):
    """Gathers a tensor from other replicas, concat on expand_dim and return."""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    return torch.cat([o.unsqueeze(expand_dim) for o in other_replica_tensors], expand_dim)


class NTXent(nn.Module):
    """Wrap a module to get self.training member."""

    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """NT-XENT Loss from SimCLR
        :param embedding1: embedding of augmentation1
        :param embedding2: embedding of augmentation2
        :param temperature: nce normalization temp
        :param num_replicas: number of compute devices
        :returns: scalar loss
        :rtype: float32
        """
        embedding1, embedding2 = embeddings[:, 0].contiguous(), embeddings[:, 1].contiguous()
        batch_size = embedding1.shape[0]
        feature_size = embedding1.shape[-1]
        num_replicas = dist.get_world_size()
        LARGE_NUM = 1e9

        if num_replicas > 1 and self.training:
            # First grab the tensor from all other embeddings
            embedding1_full = all_gather(embedding1, num_replicas=num_replicas)
            embedding2_full = all_gather(embedding2, num_replicas=num_replicas)

            # fold the tensor in to create [B, F]
            embedding1_full = embedding1_full.reshape(-1, feature_size)
            embedding2_full = embedding2_full.reshape(-1, feature_size)

            # Create pseudo-labels using the current replica id & ont-hotting
            replica_id = dist.get_rank()
            labels = torch.arange(batch_size, device=embedding1.device) + replica_id * batch_size
            labels = labels.type(torch.int64)
            full_batch_size = embedding1_full.shape[0]
            masks = F.one_hot(labels, full_batch_size).to(embedding1_full.device)
            labels = F.one_hot(labels, full_batch_size * 2).to(embedding1_full.device)
        else:  # no replicas or we are in test mode; test set is same size on all replicas for now
            embedding1_full = embedding1
            embedding2_full = embedding2
            masks = F.one_hot(torch.arange(batch_size), batch_size).to(embedding1.device)
            labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(embedding1.device)

        # Matmul-to-mask
        logits_aa = torch.matmul(embedding1, embedding1_full.T) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(embedding2, embedding2_full.T) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(embedding1, embedding2_full.T) / self.temperature
        logits_ba = torch.matmul(embedding2, embedding1_full.T) / self.temperature

        # Use our standard cross-entropy loss which uses log-softmax internally.
        # Concat on the feature dimension to provide all features for standard softmax-xent
        loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss = (loss_a + loss_b) * 0.5
        return torch.mean(loss)
