from losses import CrossEntropy, ConsistencyWeight, ContrastiveLoss, ConsistencyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import Encoder, Decoder, Projector
from torch.nn import MSELoss

class DLFCL(nn.Module):
    def __init__(self, config=None):
        super(DLFCL, self).__init__()
        self.mode = config.mode
        self.ignore_index = config.ignore_index
        self.reduction = config.reduction
        self.sup_loss_seg = CrossEntropy(self.reduction, self.ignore_index)
        self.sup_loss_sdm = MSELoss()

        self.extractor = Encoder()
        self.classifier = Decoder()

        if self.mode == 'semi':
            self.epoch_semi = config.epoch_semi
            self.in_dim = config.in_dim
            self.out_dim = config.out_dim
            self.downsample = config.downsample
            self.projector = Projector(self.in_dim, self.out_dim, self.downsample)

            self.patch_num = config.patch_num

            self.weight_sdm = config.weight_sdm
            self.weight_contr = config.weight_contr
            self.weight_cons = config.weight_cons
            self.max_epoch = config.max_epoch
            self.ramp = config.ramp

            self.contrastive_loss = ContrastiveLoss(config.temp)
            self.consistency_loss = ConsistencyLoss()
            self.get_consistency_weight = ConsistencyWeight(self.weight_cons, self.max_epoch, self.ramp)


    def forward(self, x_l=None, y_l=None, y_sdm=None, x_ul=None, epoch=None, proj_ul_ema=None, x_ul_ema=None, sdm_u_ema=None, dropout=True):

        if self.mode == 'test':

            enc = self.extractor(x_l, dropout=dropout)
            cla, sdm = self.classifier(enc)
            loss_sup1 = self.sup_loss_seg(cla, y_l)
            sup_loss_sdm = self.mse(sdm, y_sdm)

            loss_sup = loss_sup1 + self.weight_sdm * sup_loss_sdm

            return cla, sdm, loss_sup

        elif self.mode == 'train':

            enc = self.extractor(x_l, dropout=dropout)
            cla, sdm = self.classifier(enc)
            loss_sup1 = self.sup_loss_seg(cla, y_l)
            sup_loss_sdm = self.mse(sdm, y_sdm)

            loss_sup = loss_sup1 + self.weight_sdm * sup_loss_sdm

            if epoch < self.epoch_semi:
                return cla, sdm, loss_sup

            x_u = x_ul['model']
            fea_ul = self.extractor(x_u, dropout=dropout)
            proj_ul = self.projector(fea_ul)  # [b, c, h, w]
            proj_ul = F.normalize(proj_ul, 2, 1)

            cla_u, sdm_u = self.classifier(fea_ul)

            max_probs = torch.softmax(cla, 1)
            max_probs_ema = torch.softmax(x_ul_ema, 1)

            consistency_weight = self.get_consistency_weight(epoch)
            loss_cons = self.consistency_loss(max_probs, max_probs_ema)
            loss_cons = torch.mean(loss_cons)
            sdm_u = sdm_u.detach()
            sdm_u_ema = sdm_u_ema.detach()

            b, c = proj_ul.size(0), proj_ul.size(1)
            h, w = proj_ul.size(2) // self.patch_num, proj_ul.size(3) // self.patch_num

            h_l, w_l = h * 4, w * 4

            patches = []
            patches_ema = []
            patch_labels = []
            patch_labels_ema = []
            FC = []
            FC_ema = []
            # nowk = 0
            for i in range(self.patch_num * self.patch_num):
                j = i // self.patch_num
                k = i % self.patch_num
                for ii in range(b):

                    p = proj_ul[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    p_ema = proj_ul_ema[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]

                    pla = sdm_u[ii, 0, j * h_l: (j + 1) * h_l, k * w_l: (k + 1) * w_l]
                    pla_ema = sdm_u_ema[ii, 0, j * h_l: (j + 1) * h_l, k * w_l: (k + 1) * w_l]
                    patches.append(p)
                    patches_ema.append(p_ema)
                    patch_labels.append(pla)
                    patch_labels_ema.append(pla_ema)

                    fc = pla.sum().item()
                    fc_ema = pla_ema.sum().item()

                    FC.append([fc] * h * w)
                    FC_ema.append([fc_ema] * h * w)

            _patches = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches]
            _patches_ema = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches_ema]
            _patches = torch.cat(_patches, 0)
            _patches_ema = torch.cat(_patches_ema, 0)
            _patch_labels = [p.contiguous().view(-1) for p in patch_labels]
            _patch_labels_ema = [p.contiguous().view(-1) for p in patch_labels_ema]
            _patch_labels = torch.cat(_patch_labels, 0)
            _patch_labels_ema = torch.cat(_patch_labels_ema, 0)
            _patches = torch.cat([_patches, _patches_ema], 0)
            _patch_labels = torch.cat([_patch_labels, _patch_labels_ema], 0)
            _FC = torch.cat([torch.tensor(FC), torch.tensor(FC_ema)], 0).view(-1).cuda()

            feature_all = _patches
            FC_all = _FC

            loss_contr_sum = 0.0
            loss_contr_count = 0

            for i_patch in range(b * self.patch_num * self.patch_num):
                patch_i = patches[i_patch]
                patch_ema_i = patches_ema[i_patch]
                pseudo_label_i = patch_labels[i_patch]

                patch_i = patch_i.permute(1, 2, 0).contiguous().view(-1, proj_ul.size(1))
                patch_i_ema = patch_ema_i.permute(1, 2, 0).contiguous().view(-1, proj_ul_ema.size(1))
                pseudo_label_i = pseudo_label_i.contiguous().view(-1)

                fc = pseudo_label_i.sum()

                loss_contr_count += 1
                FC_i = [fc] * h * w
                FC_i = torch.tensor(FC_i).cuda()

                loss_contr = torch.utils.checkpoint.checkpoint(self.contrastive_loss, patch_i, patch_i_ema,
                                                               feature_all, FC_i, FC_all)

                loss_contr = loss_contr.mean()
                loss_contr_sum += loss_contr

            loss_contr = loss_contr_sum / loss_contr_count

            loss_total = loss_sup + self.weight_contr * loss_contr + consistency_weight * loss_cons

            return cla, sdm, loss_total

