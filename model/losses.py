import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.extractor.parameters(), model.extractor.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(ema_model.projector.parameters(), model.projector.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(ema_model.classifier.parameters(), model.classifier.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        _targets = targets.clone()
        if mask is not None:
            _targets[mask] = self.ignore_index

        loss = F.cross_entropy(inputs, _targets, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='mean')



class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    # patch_i, patch_i_ema, feature_all, pseudo_label_i, FC_i, FC_all
    def forward(self, input_logits, target_logits):

        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        mse_loss = (input_softmax - target_softmax) ** 2
        return mse_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    # patch_i, patch_i_ema, feature_all, pseudo_label_i, FC_i, FC_all
    def forward(self, anchor, pos_pair, neg_pair, FC, FC_all):

        pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True)
        now = FC.unsqueeze(-1)[0]

        if now >= 0:
            mask_patch_filter = (FC_all.unsqueeze(0) < 0).float()
        else:
            mask_patch_filter = (FC_all.unsqueeze(0) >= 0).float()

        mask_patch_filter = torch.cat([torch.ones(mask_patch_filter.size(0), 1).float().cuda(), mask_patch_filter], 1)

        neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp)
        neg = torch.cat([pos, neg], 1)
        max = torch.max(neg, 1, keepdim=True)[0]
        #
        exp_neg = (torch.exp(neg - max) * mask_patch_filter).sum(-1)

        loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps)
        loss = -torch.log(loss + self.eps)

        return loss


class ConsistencyWeight(nn.Module):
    def __init__(self, max_weight, max_epoch, ramp='sigmoid'):
        super(ConsistencyWeight, self).__init__()
        self.max_weight = max_weight
        self.max_epoch = max_epoch
        self.ramp = ramp

    def forward(self, epoch):
        current = np.clip(epoch, 0.0, self.max_epoch)
        phase = 1.0 - current / self.max_epoch
        if self.ramp == 'sigmoid':
            ramps = float(np.exp(-5.0 * phase * phase))
        elif self.ramp == 'log':
            ramps = float(1 - np.exp(-5.0 * current / self.max_epoch))
        elif self.ramp == 'exp':
            ramps = float(np.exp(5.0 * (current / self.max_epoch - 1)))
        else:
            ramps = 1.0

        consistency_weight = self.max_weight * ramps
        return consistency_weight
