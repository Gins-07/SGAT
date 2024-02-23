import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.base import BaseModel
from Model.utils.helpers import initialize_weights
from itertools import chain

class InitalBlock_enet(nn.Module):
    def __init__(self, in_channels, use_prelu=True):
        super(InitalBlock_enet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16) if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x):   
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class BottleNeck_enet(nn.Module):
    def __init__(self, in_channels, out_channels=None, activation=None, dilation=1, downsample=False, proj_ratio=4, 
                        upsample=False, asymetric=False, regularize=True, p_drop=None, use_prelu=True):
        super(BottleNeck_enet, self).__init__()

        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        if out_channels is None: out_channels = in_channels
        else: self.pad = out_channels - in_channels

        if regularize: assert p_drop is not None
        if downsample: assert not upsample
        elif upsample: assert not downsample
        inter_channels = in_channels//proj_ratio

        # Main
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # BottleNeck_enet
        if downsample: 
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        if asymetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1,5), padding=(0,2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU() if use_prelu else nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(5,1), padding=(2,0)),
            )
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x
        if self.upsample:
            assert (indices is not None) and (output_size is not None)
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = (indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0)
                identity = F.pad(identity, pad, "constant", 0)
            identity = self.unpool(identity, indices=indices)#, output_size=output_size)
        elif self.downsample:
            identity, idx = self.pool(identity)

        '''
        if self.pad > 0:
            if self.pad % 2 == 0 : pad = (0, 0, 0, 0, self.pad//2, self.pad//2)
            else: pad = (0, 0, 0, 0, self.pad//2, self.pad//2+1)
            identity = F.pad(identity, pad, "constant", 0)
        '''

        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # BottleNeck_enet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu_out(x)

        if self.downsample:
            return x, idx
        return x

class ENet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ENet, self).__init__()
        self.initial = InitalBlock_enet(in_channels)

        # Stage 1
        self.BottleNeck_enet10 = BottleNeck_enet(16, 64, downsample=True, p_drop=0.01)
        self.BottleNeck_enet11 = BottleNeck_enet(64, p_drop=0.01)
        self.BottleNeck_enet12 = BottleNeck_enet(64, p_drop=0.01)
        self.BottleNeck_enet13 = BottleNeck_enet(64, p_drop=0.01)
        self.BottleNeck_enet14 = BottleNeck_enet(64, p_drop=0.01)

        # Stage 2
        self.BottleNeck_enet20 = BottleNeck_enet(64, 128, downsample=True, p_drop=0.1)
        self.BottleNeck_enet21 = BottleNeck_enet(128, p_drop=0.1)
        self.BottleNeck_enet22 = BottleNeck_enet(128, dilation=2, p_drop=0.1)
        self.BottleNeck_enet23 = BottleNeck_enet(128, asymetric=True, p_drop=0.1)
        self.BottleNeck_enet24 = BottleNeck_enet(128, dilation=4, p_drop=0.1)
        self.BottleNeck_enet25 = BottleNeck_enet(128, p_drop=0.1)
        self.BottleNeck_enet26 = BottleNeck_enet(128, dilation=8, p_drop=0.1)
        self.BottleNeck_enet27 = BottleNeck_enet(128, asymetric=True, p_drop=0.1)
        self.BottleNeck_enet28 = BottleNeck_enet(128, dilation=16, p_drop=0.1)
    
        # Stage 3
        self.BottleNeck_enet31 = BottleNeck_enet(128, p_drop=0.1)
        self.BottleNeck_enet32 = BottleNeck_enet(128, dilation=2, p_drop=0.1)
        self.BottleNeck_enet33 = BottleNeck_enet(128, asymetric=True, p_drop=0.1)
        self.BottleNeck_enet34 = BottleNeck_enet(128, dilation=4, p_drop=0.1)
        self.BottleNeck_enet35 = BottleNeck_enet(128, p_drop=0.1)
        self.BottleNeck_enet36 = BottleNeck_enet(128, dilation=8, p_drop=0.1)
        self.BottleNeck_enet37 = BottleNeck_enet(128, asymetric=True, p_drop=0.1)
        self.BottleNeck_enet38 = BottleNeck_enet(128, dilation=16, p_drop=0.1)

        # Stage 4
        self.BottleNeck_enet40 = BottleNeck_enet(128, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.BottleNeck_enet41 = BottleNeck_enet(64, p_drop=0.1, use_prelu=False)
        self.BottleNeck_enet42 = BottleNeck_enet(64, p_drop=0.1, use_prelu=False)

        # Stage 5
        self.BottleNeck_enet50 = BottleNeck_enet(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.BottleNeck_enet51 = BottleNeck_enet(16, p_drop=0.1, use_prelu=False)

        # Stage 6
        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        initialize_weights(self)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        x = self.initial(x)

        # Stage 1
        sz1 = x.size()
        x, indices1 = self.BottleNeck_enet10(x)
        x = self.BottleNeck_enet11(x)
        x = self.BottleNeck_enet12(x)
        x = self.BottleNeck_enet13(x)
        x = self.BottleNeck_enet14(x)

        # Stage 2
        sz2 = x.size()
        x, indices2 = self.BottleNeck_enet20(x)
        x = self.BottleNeck_enet21(x)
        x = self.BottleNeck_enet22(x)
        x = self.BottleNeck_enet23(x)
        x = self.BottleNeck_enet24(x)
        x = self.BottleNeck_enet25(x)
        x = self.BottleNeck_enet26(x)
        x = self.BottleNeck_enet27(x)
        x = self.BottleNeck_enet28(x)

        # Stage 3
        x = self.BottleNeck_enet31(x)
        x = self.BottleNeck_enet32(x)
        x = self.BottleNeck_enet33(x)
        x = self.BottleNeck_enet34(x)
        x = self.BottleNeck_enet35(x)
        x = self.BottleNeck_enet36(x)
        x = self.BottleNeck_enet37(x)
        x = self.BottleNeck_enet38(x)

        # Stage 4
        x = self.BottleNeck_enet40(x, indices=indices2, output_size=sz2)
        x = self.BottleNeck_enet41(x)
        x = self.BottleNeck_enet42(x)

        # Stage 5
        x = self.BottleNeck_enet50(x, indices=indices1, output_size=sz1)
        x = self.BottleNeck_enet51(x)

        # Stage 6
        x = self.fullconv(x)
        return x 

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

