import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class DeepMutualLoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w if w > 0  else -w
        self.T = temperature

        self.neg = w < 0

    def forward(self, logits, targets):
        n = len(logits)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets) for i in range(n)]
        ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)

        # KD Loss
        kd_loss = [1. / (n-1) * 
                   self.kd_criterion(
                       F.log_softmax(logits[i] / self.T, dim=1), 
                       F.log_softmax(logits[j] / self.T, dim=1).detach()
                   ) * self.T * self.T
                   for i, j in itertools.permutations(range(n), 2)]
        kd_loss = torch.sum(torch.stack(kd_loss, dim=0), dim=0)
        if self.neg:
            kd_loss = -1.0 * kd_loss

        total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()


class ONELoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w
        self.T = temperature

    def forward(self, logits, targets):
        n = len(logits)
        ensemble_logits = torch.mean(torch.stack(logits, dim=0), dim=0)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets) for i in range(n)] + [self.base_criterion(ensemble_logits, targets)]
        #ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)
        ce_loss = torch.mean(torch.stack(ce_loss, dim=0), dim=0)

        # One Loss
        kd_loss = [self.kd_criterion(
            F.log_softmax(logits[i] / self.T, dim=1), 
            F.log_softmax(ensemble_logits / self.T, dim=1).detach()
        ) * self.T * self.T for i in range(n)]
        #kd_loss = torch.sum(torch.stack(kd_loss, dim=0), dim=0)
        kd_loss = torch.mean(torch.stack(kd_loss, dim=0), dim=0)

        #total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        #total_loss = (1.0 - self.w) * ce_loss - self.w * kd_loss
        total_loss = ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()


class MulMixLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(MulMixLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target, beta=1.0):
        inv_prob = torch.pow(1.0 - F.softmax(x, dim=-1), beta)
        logprobs = F.log_softmax(x, dim=-1)
        logprobs = logprobs * inv_prob
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MulMixSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(MulMixSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, beta=1.0):
        inv_prob = torch.pow(1.0 - F.softmax(x, dim=-1), beta)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1) * inv_prob, dim=-1)
        return loss.mean()


class MulMixturelLoss(nn.Module):

    def __init__(self, base_criterion, beta):
        super().__init__()

        if isinstance(base_criterion, LabelSmoothingCrossEntropy):
            self.base_criterion = MulMixLabelSmoothingCrossEntropy(base_criterion.smoothing)
        elif isinstance(base_criterion, SoftTargetCrossEntropy):
            self.base_criterion = MulMixSoftTargetCrossEntropy()
        else:
            raise ValueError("Unknown type")
            
        self.beta = beta

    def forward(self, logits, targets):
        n = len(logits)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets, self.beta / (n - 1)) for i in range(n)]
        ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)
        return ce_loss


class SelfDistillationLoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w
        self.T = temperature

    def forward(self, logits, targets):
        # logits is a list, the first one is the reference logits for self-distillation

        # CE losses
        ce_loss = self.base_criterion(logits[1], targets)

        # KD Loss
        kd_loss = self.kd_criterion(
            F.log_softmax(logits[1] / self.T, dim=1),
            F.log_softmax(logits[0] / self.T, dim=1).detach()
        ) * self.T * self.T

        total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()
