from typing import List

import torch
import torch.nn as nn
from torch import Tensor, einsum
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from typing import List


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])



class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
        print("test")

    #def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        if probs.shape[1] ==1:
            probs = torch.nn.functional.sigmoid(probs)
            probs = torch.cat((1-probs,probs),1)
            dist_maps = torch.cat((1-dist_maps,dist_maps),1)
#         if not simplex(probs):
#             print(probs.shape,probs.max(),probs.min())
#         if one_hot(dist_maps):
#             print(dist_maps.shape,dist_maps.max())
        #assert simplex(probs)
        #assert not one_hot(dist_maps)
        #print(probs.shape,dist_maps.shape)
        
        #pc = probs[:, self.idc, ...].type(torch.float32)
        #dc = dist_maps[:, self.idc, ...].type(torch.float32)
        pc = probs[:, ...].type(torch.float32)
        dc = dist_maps[:, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


# class DiceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         #self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized")

#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         #assert simplex(probs) and simplex(target)
#         if probs.shape[1] ==1:
#             probs = torch.nn.functional.sigmoid(probs)
#             probs = torch.cat((1-probs,probs),1)
#             target = torch.cat((1-target,target),1)
            
#         pc = probs[:, ...].type(torch.float32)
#         tc = target[:, ...].type(torch.float32)

#         intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
#         union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

#         divided: Tensor = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

#         loss = divided.mean()

#         return loss

class dicelossmulticlass():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        print(f"Initialized")

    def __call__(self, probs: Tensor, target: Tensor,eps=1e-7) -> Tensor:
        #assert simplex(probs) and simplex(target)
        num_classes = probs.shape[1]
#         print(num_classes, probs.shape)
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1).to(torch.int64)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(probs)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
#             print(target.squeeze(1).shape)
            true_1_hot = torch.eye(num_classes)[target.squeeze(1).to(torch.int64)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(probs, dim=1)
        true_1_hot = true_1_hot.type(probs.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
        print("test")

    #def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        if probs.shape[1] ==1:
            probs = torch.nn.functional.sigmoid(probs)
            probs = torch.cat((1-probs,probs),1)
            dist_maps = torch.cat((1-dist_maps,dist_maps),1)
#         if not simplex(probs):
#             print(probs.shape,probs.max(),probs.min())
#         if one_hot(dist_maps):
#             print(dist_maps.shape,dist_maps.max())
        #assert simplex(probs)
        #assert not one_hot(dist_maps)
        #print(probs.shape,dist_maps.shape)
        
        #pc = probs[:, self.idc, ...].type(torch.float32)
        #dc = dist_maps[:, self.idc, ...].type(torch.float32)
        pc = probs[:, ...].type(torch.float32)
        dc = dist_maps[:, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
#         tmp_targets = targets.view(-1,1,1)
#         print(inputs.shape,targets.shape)
        if targets.is_cuda:
            tmp_device = targets.get_device()
#             print("targets.size = " + str(targets.size()))
            if  len(targets.size())==2 or  len(targets.size())==1:
                tmp_targets = torch.ones(size=(inputs.shape[0],inputs.shape[2],inputs.shape[3]),device=torch.device(tmp_device))*(targets.view(-1,1,1).float())
            elif len(targets.size())==3:
                tmp_targets = targets.to(tmp_device)
            elif len(targets.size())==4:
                tmp_targets = targets.view(-1,targets.shape[2],targets.shape[2]).to(tmp_device)
            else:
                raise RuntimeError('dimension Error')
        else:
            if  len(targets.size())==2:
                tmp_targets = torch.ones(size=(inputs.shape[0],inputs.shape[2],inputs.shape[3]))*(targets.view(-1,1,1))
            elif len(targets.size())==3:
                pass
            else:
                raise RuntimeError('dimension Error')
        tmp_targets = tmp_targets.long()
        return self.nll_loss(torch.nn.functional.log_softmax(inputs), tmp_targets)
    

class CrossEntropyLoss2d_unet(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d_unet, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.nn.functional.log_softmax(inputs, dim=1), targets.long())

    
class CrossEntropyLoss2d_1dim(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d_1dim, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs.view(-1,inputs.shape[2],inputs.shape[3]) , targets.view(-1,inputs.shape[2],inputs.shape[3]).long(),)
    
class MultiTaskLoss(nn.Module):
    """
    multi-task loss: cls seg
    """

    def __init__(self, seg_loss, cls_loss):
        super(MultiTaskLoss, self).__init__()
        self.seg_loss = seg_loss
        self.cls_loss = cls_loss

    def forward(self, seg_output, mask, cls_output, label):
#         if seg_output is not None:
        seg_losses = self.seg_loss(seg_output, mask)
#         if cls_output is not None:
        cls_losses = self.cls_loss(cls_output, label)
        
        return seg_losses + cls_losses
    
class MultiLoss(nn.Module):
    """
    multi loss: cls seg
    """

    def __init__(self, losses = [], names = [], weights = []):
        super(MultiLoss, self).__init__()
        self.losses = losses
        self.names = names
        self.weights = weights

    def forward(self, seg_output, mask):
        loss_dict = {}
        total_loss = 0
        for i, loss in enumerate(self.losses):
            loss_value = loss(seg_output, mask)
            total_loss+=(loss_value*self.weights[i])
            key = self.names[i]
            loss_dict[key]=loss_value
        
        loss_dict['total_loss']=total_loss
        return loss_dict

    
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
# #         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice
    
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1., ignore_index=255):
#         super(DiceLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth

#     def forward(self, output, target):
#         if self.ignore_index not in range(target.min(), target.max()):
#             if (target == self.ignore_index).sum() > 0:
#                 target[target == self.ignore_index] = target.min()
#         target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
#         output = F.softmax(output, dim=1)
#         output_flat = output.contiguous().view(-1)
#         target_flat = target.contiguous().view(-1)
#         intersection = (output_flat * target_flat).sum()
#         loss = 1 - ((2. * intersection + self.smooth) /
#                     (output_flat.sum() + target_flat.sum() + self.smooth))
#         return loss
    
# def _iou(pred, target, size_average = True):

#     b = pred.shape[0]
#     IoU = 0.0
#     for i in range(0,b):
#         #compute the IoU of the foreground
#         Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
#         Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
#         IoU1 = Iand1/Ior1

#         #IoU loss is (1-IoU1)
#         IoU = IoU + (1-IoU1)
# #     print(IoU/b)
#     return IoU/b

# class IoULoss(torch.nn.Module):
#     def __init__(self, label = 1, size_average = True):
#         super(IoULoss, self).__init__()
#         self.size_average = size_average
#         self.category = label

#     def forward(self, pred, target):
# #         target = target.unsqueeze(1) # N,1,H,W
# #         print(target.shape, pred.shape)
# #         pred = pred[:,self.category,:,:].unsqueeze(1) # N,C,H,W -> N,1,H,W
#         pred = torch.exp(pred) # convert log_softmax to softmax
#         target = (target==self.category)

#         return _iou(pred, target, self.size_average)
    
# class IoULoss(nn.Module):
#     def __init__(self, label = 1, weight=None, size_average=True):
#         super(IoULoss, self).__init__()
#         self.category = label

#     def forward(self, inputs, targets, smooth=1):
#         if inputs.dim()>2:
#             inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
#             inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
#             inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
#         targets = targets.view(-1,1)
#         inputs = inputs.gather(1,targets)
              
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         targets = (targets==self.category)
        
#         #intersection is equivalent to True Positive count
#         #union is the mutually inclusive area of all labels & predictions 
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection 
        
#         IoU = (intersection + smooth)/(union + smooth)
        
#         return 1 - IoU
    
def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22
        
        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou

class GeneralizedL1Loss(nn.Module):
    '''
    l = at * \alpha * |p-p*|** \gamma
    input (Tensor): N,C or N,C,H,W
    beta: balance
    '''
    def __init__(self, alpha=1, gamma=1, beta = None, size_average=True):
        super(GeneralizedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        if isinstance(beta,(float,int)): self.beta = torch.Tensor([beta,1-beta])
        if isinstance(beta,list): self.beta = torch.Tensor(beta)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        p = input.gather(1,target)
        p = p.view(-1)
        if self.beta is not None:
            if self.beta.type()!=input.data.type():
                self.beta = self.beta.type_as(input.data)
            bt = self.beta.gather(0,target.data.view(-1))
            losses = Variable(bt) * self.alpha * (1-p)**self.gamma
            return losses.mean() if self.size_average else losses.sum()
        else:
            losses = self.alpha * (1-p)**self.gamma  # torch.abs(p - 1)
            return losses.mean() if self.size_average else losses.sum()

# class FocalLoss(nn.Module):
#     '''
#     l = - \at * log pt * |1-pt|** \gamma 
    
#     '''
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1

# #         logpt = F.log_softmax(input, dim=-1)
#         logpt = torch.log(input)
#         logpt = logpt.gather(1,target) # gather confidence of gt class based on target
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         losses = -1 * (1-pt)**self.gamma * logpt
#         return losses.mean() if self.size_average else losses.sum()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    output1/output2: embeddings nx2
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        # self.dice = DiceLoss()
        self.dice = dicelossmulticlass()
#         self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self.cross_entropy =  CrossEntropyLoss2d_cedice()
    
    def forward(self, output, target):
#         print(output.shape, target.shape)
        CE_loss = self.cross_entropy(output, (target).long())
        dice_loss = self.dice(output[:,-1:], target)
        return CE_loss + dice_loss
    
    
class CrossEntropyLoss2d_cedice(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d_cedice, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
#         tmp_targets = targets.view(-1,1,1)
#         print(inputs.shape,targets.shape)
        if targets.is_cuda:
            tmp_device = targets.get_device()
#             print("targets.size = " + str(targets.size()))
            if len(targets.size())==3:
                tmp_targets = targets.to(tmp_device)
            elif len(targets.size())==4:
                tmp_targets = targets.view(-1,targets.shape[2],targets.shape[2]).to(tmp_device)
            else:
                raise RuntimeError('dimension Error')
        tmp_targets = tmp_targets.long()
        return self.nll_loss(torch.nn.functional.log_softmax(inputs), tmp_targets)
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        target = target.view(-1,target.shape[2],target.shape[2])
        logpt = self.CE_loss(output, target.long())
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()