import torch
import torch.nn as nn
import torch.nn.functional as F
###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)   #kernel_size=31*31
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)



###################################################################
# #################### structure loss #############################
###################################################################
class uncertainty_aware_structure_loss(torch.nn.Module):
    def __init__(self):
        super(uncertainty_aware_structure_loss, self).__init__()

    def _uncertainty_aware_structure_loss(self, pred, mask, confi_map, epoch, f1=1, f2=10, epsilon=1):
        if epoch < 20:
            f2 = 0
        weit = 1 + f1 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + f2 * confi_map
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
        return (wbce + wiou).mean()

    def forward(self, pred, mask,confi_map,epoch):
        return self._uncertainty_aware_structure_loss(pred, mask,confi_map,epoch)


###################################################################
# #################### CCAM loss #############################
###################################################################
def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        return loss

        # if self.reduction == 'mean':
        #     return torch.mean(loss)
        # elif self.reduction == 'sum':
        #     return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        # self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)


            # loss[loss < 0] = 0
            # _, indices = sim.sort(descending=True, dim=1)
            # _, rank = indices.sort(dim=1)
            # rank = rank - 1
            # rank_weights = torch.exp(-rank.float() * self.alpha)
            # loss = loss * rank_weights
        else:
            raise NotImplementedError

        # if self.reduction == 'mean':
        #     return torch.mean(loss)
        # elif self.reduction == 'sum':
        #     return torch.sum(loss)
        return loss


def make_confidence_label(pred, gts):
    """
    :param pred: COD prediction
    :param gts: COD GT
    :return: OCE_Net supervision
    """
    C_label = (torch.mul(gts, (1 - pred)) + torch.mul((1 - gts), pred))
    return C_label
