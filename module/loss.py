import torch.nn.functional as F
import torch.nn as nn
import torch
import ever as er
from ever.interface.learning_rate import LearningRateBase
from simplecv.module.loss import dice_loss_with_logits

# blow yh's conde
def _iou_1(y_true, y_pred,ignore_index=255):
    with torch.no_grad():
        y_pred = (y_pred.argmax(dim=1)).float().reshape(-1)
        y_true = y_true.float().reshape(-1)
        mask = y_true == ignore_index
        valid = ~mask
        y_true = y_true.masked_select(valid).float()
        y_pred = y_pred.masked_select(valid).float()
        inter = torch.sum(y_pred * y_true)
        union = y_true.sum() + y_pred.sum()
        return inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))


def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    total_loss = torch.mean(total_loss)
    return total_loss / num_classes


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


# up yh's conde
#def kl_div(out1, out2):
#    loss = F.kl_div(F.log_softmax(out1, dim=1),
#                    F.softmax(out2, dim=1),
#                    reduction='mean') * out1.shape[1]
#    return loss

def kl_div(out1, out2,T=1):
    h,w=out1.shape[2],out1.shape[3]
    loss = F.kl_div(F.log_softmax(out1/T, dim=1),
    F.softmax(out2/T, dim=1),
    reduction='batchmean')
    return loss/(h*w)
def kl_div_sigmoid(out1, out2,T=1):
    h,w=out1.shape[2],out1.shape[3]
    loss = F.kl_div(F.logsigmoid(out1/T),
    F.sigmoid(out2/T),
    reduction='mean')
    return loss

def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int=-1):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true
def infoNCE_loss(m1_pred: torch.Tensor, m2_pred,y_true: torch.Tensor = None, ignore_index: int=-1):
    _,C,_,_ = m1_pred.shape
    assert m1_pred.shape == m2_pred.shape
    if m1_pred.shape[2] > 30:
        m1_pred = F.adaptive_avg_pool2d(m1_pred, (32, 32))
        m2_pred = F.adaptive_avg_pool2d(m2_pred, (32, 32))
    if y_true != None:
        y_true = y_true.squeeze()
        y_true = y_true.reshape(-1)
    m1_pred = m1_pred.permute((0, 2, 3, 1)) 
    m2_pred = m2_pred.permute((0, 2, 3, 1))
    m1_pred = m1_pred.reshape(-1,C)
    m2_pred = m2_pred.reshape(-1,C)
    if y_true != None:
        mask = y_true != ignore_index
        m1_pred = m1_pred[mask].contiguous()
        m2_pred = m2_pred[mask].contiguous()
    logit = torch.mm(m1_pred,m2_pred.t())
    label = torch.eye(m1_pred.shape[0]).to(logit.device)
    return nn.BCEWithLogitsLoss()(logit, label)

def tversky_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float,
                             smooth_value: float = 1.0,
                             ignore_index: int = -1):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1. - tversky_coeff

def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                                     ignore_index: int = -1):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)

def _iou_1_sigmoid(y_true, y_pred,ignore_index=255):
    with torch.no_grad():
        y_pred = (y_pred.sigmoid() > 0.5).float().reshape(-1)
        y_true = y_true.float().reshape(-1)
        inter = torch.sum(y_pred * y_true)
        union = y_true.sum() + y_pred.sum()
        return inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))


def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses


class SegmentationLoss(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLoss, self).__init__()
        self.loss_config = loss_config

    def forward(self, y_pred, y_true: torch.Tensor,pre = '',Lamada=1):
        loss_dict = dict()
        # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)
        
        #blow 'yh's code'
        # up 'yh's code'
        if 'bce' in self.loss_config:
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            mask = y_true == self.loss_config.ignore_index
            valid = ~mask
            y_true = y_true.masked_select(valid).float()
            y_pred = y_pred.masked_select(valid).float()
            if 'ohem' in self.loss_config.bce:
                loss_dict['bce_loss'] = ohem(F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none'),
                                             self.loss_config.bce.ohem.ratio)
            else:
                loss_dict['bce_loss'] = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
        if "iousigmoid" in self.loss_config:
            loss_dict['iou-1'] = _iou_1_sigmoid(y_true, y_pred,ignore_index=self.loss_config.ignore_index)
        if "iou" in self.loss_config:
            loss_dict['iou-1'] = _iou_1(y_true, y_pred,ignore_index=self.loss_config.ignore_index)
        if 'dice' in self.loss_config:
            loss_dict['dice_loss'] = dice_loss_with_logits(y_pred, y_true, ignore_index=self.loss_config.ignore_index)
        if 'ce' in self.loss_config:
            loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.loss_config.ignore_index)
        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config.fcloss.gamma, normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true, self.loss_config.ignore_index) * self.loss_config.bceloss.scaler


        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config.tverloss.alpha,
                                                             self.loss_config.tverloss.beta, ignore_index=-1) * self.loss_config.tverloss.scaler
        if pre:
            loss_dict = {pre + k: v for k, v in loss_dict.items()}
        loss_dict = {k: v*Lamada for k, v in loss_dict.items()}
        return loss_dict

def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, ):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()



def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true



# def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
#                           smooth_value: float = 1.0,
#                           ignore_index: int = -1,
#                           ignore_channel: int = -1):
#     c = y_pred.size(1)
#     y_pred, y_true = select(y_pred, y_true, ignore_index)
#     weight = torch.as_tensor([True] * c, device=y_pred.device)
#     if c == 1:
#         y_prob = y_pred.sigmoid()
#         return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
#     else:
#         y_prob = y_pred.softmax(dim=1)
#         y_true = F.one_hot(y_true, num_classes=c)
#         if ignore_channel != -1:
#             weight[ignore_channel] = False
# 
#         return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)

def label_smoothing_dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, eps:float = 0.1,
                          smooth_value: float = 1.0,
                          ignore_index: int = -1,
                          ignore_channel: int = -1,  
                          ):
                          
    y_true = torch.where(y_true == 0, y_true + eps, y_true - eps)
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)


def multi_binary_label(batched_mask: torch.Tensor, num_classes):
    labels = []
    for cls in range(1, num_classes+1):
        labels.append((batched_mask == cls).to(torch.long))
    return labels

def label_smoothing_binary_cross_entropy(output: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                                         reduction: str = 'mean', ignore_index: int = -1):
    output, target = _masked_ignore(output, target, ignore_index)
    target = torch.where(target == 0, target + eps, target - eps)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)




def multi_binary_loss(y_pred, y_true, num_classes, reduction='mean', label_smooth=0., dice_scaler=0., bce_scaler=0.):
    labels = multi_binary_label(y_true, num_classes)
    losses = []
    for cls in range(num_classes):
        bipred = y_pred[:, cls, :, :]
        bipred = bipred.reshape(y_pred.size(0), 1, y_pred.size(2), y_pred.size(3)).contiguous()
        if label_smooth > 0:
            bce_loss = label_smoothing_binary_cross_entropy(bipred, labels[cls].reshape_as(bipred).float(),
                                                                     label_smooth)
            dice_loss = label_smoothing_dice_loss_with_logits(bipred, labels[cls], label_smooth)
        else:
            bce_loss = binary_cross_entropy_with_logits(bipred, labels[cls].reshape_as(bipred).float(), ignore_index=-1)
            dice_loss = dice_loss_with_logits(bipred, labels[cls], ignore_index=-1)
        losses.append(bce_loss*bce_scaler+dice_loss*dice_scaler)

    if 'sum' == reduction:
        tloss = sum(losses)
    elif 'mean' == reduction:
        tloss = sum(losses) / float(num_classes)
    else:
        raise ValueError()

    if label_smooth > 0:
        return dict(mb_sm_loss=tloss)
    else:
        return dict(mb_loss=tloss)
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
@er.registry.LR.register('const')
class ConstLearningRate(LearningRateBase):
    def __init__(self,
                 base_lr,
                 ):
        super(ConstLearningRate, self).__init__(base_lr)

    def step(self, global_step, optimizer):
        set_lr(optimizer, self.base_lr)


