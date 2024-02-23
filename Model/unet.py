import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# attention module
# from Model.bam import BAM
from Model.cbam import CBAM
# from Model.se import SELayer
from Model.sk import SKConv

class skconv_block_yb_yb(nn.Module):
    def __init__(self,ch_in,ch_out, WH=None, M=2, G=1, r=8, stride=1, L=32, is_shortcut = False, reduction_ratio = None, att_mode = 'cbam'):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(skconv_block_yb_yb, self).__init__()
        self.feas = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SKConv(ch_out, WH, M, G, r, stride=stride, L=L)
        )
        self.is_shortcut = is_shortcut
        if self.is_shortcut:
            if ch_in == ch_out: # when dim not change, in could be added diectly to out， Identity Shortcut
                self.shortcut = nn.Sequential()
            else: # when dim not change, in should also change dim to be added to out， Projection Shortcut
                self.shortcut = nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 1, stride=stride),
                    nn.BatchNorm2d(ch_out)
                )
            
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention
    
    def forward(self, x):
        out = self.feas(x)
        if self.is_shortcut:
            out = out + self.shortcut(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class conv_block_yb(nn.Module):
    def __init__(self,ch_in,ch_out, reduction_ratio = None, att_mode = None):
        super(conv_block_yb,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention


    def forward(self,x):
        out = self.conv(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class encoder_block_yb_v1(nn.Module):
    '''down_conv
    '''

    def __init__(self, ch_in, ch_out, reduction_ratio, dropout=False, first_block = False, att_mode = 'cbam', conv_type = 'basic'):
        super(encoder_block_yb_v1, self).__init__()
        if conv_type == 'basic':
            conv = conv_block_yb(ch_in,ch_out, reduction_ratio, att_mode)
        elif conv_type == 'sk':
            conv = skconv_block_yb_yb(ch_in,ch_out, reduction_ratio = reduction_ratio, att_mode = att_mode)
            
        if first_block:
            layers = [conv]
        else:
            layers = [
                nn.MaxPool2d(2, stride=2),
                conv
            ]
        if dropout:
            layers += [nn.Dropout(.2)]
        
        self.down = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention

        
    def forward(self, x):
        out = self.down(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class encoder_block_yb(nn.Module):
    '''down_conv
    '''

    def __init__(self, ch_in, ch_out, reduction_ratio, dropout=False, first_block = False, att_mode = 'cbam'):
        super(encoder_block_yb, self).__init__()
        if first_block:
            layers = [
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)        
            ]
        else:
            layers = [
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            ]
        if dropout:
            layers += [nn.Dropout(.2)]

        self.down = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention

        
    def forward(self, x):
        out = self.down(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class decoder_block_yb(nn.Module):
    '''up_conv
    '''

    def __init__(self, ch_in, ch_out, reduction_ratio = None, att_mode = 'cbam'):
        super(decoder_block_yb, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention

    def forward(self, x):
        out = self.up(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class head_block(nn.Module):
    '''up_conv
       ch_out: C
    '''

    def __init__(self, ch_in, ch_out, scale_factor, reduction_ratio = None, att_mode = 'cbam'):
        super(head_block, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            if att_mode == 'bam':
                self.att_module = BAM(ch_out, self.reduction_ratio) # BAM: Bottleneck Attention Module (BMVC2018)
            elif att_mode == 'cbam':
                self.att_module = CBAM(ch_out, self.reduction_ratio) # Convolutional Block Attention Module, channel and spatial attention
            elif att_mode == 'se':
                self.att_module = SELayer(ch_out, self.reduction_ratio) # Squeeze-and-Excitation, channel attention

    def forward(self, x):
        out = self.up(x)
        if self.reduction_ratio:
            scale_weight = self.att_module(out)
            out = out * scale_weight.expand_as(out)
        return out
    
class cbam(nn.Module):
    '''reduction_ratio = None -> U-NET
    '''

    def __init__(self, img_ch=3, n_classes=2, init_features=32, network_depth=5, reduction_ratio=None, n_skip=4, n_head = 3, att_mode = 'cbam', is_head_selective = False, is_shortcut = False, conv_type = 'basic', activation = nn.LogSoftmax(dim=1)):
        super(cbam, self).__init__()
        self.n_classes = n_classes
        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.n_skip = n_skip # [1/16,1/8,1/4,1/2,1] [0,1,2,3,4], must <=self.network_depth-1
        self.n_head = n_head
        
        decoder_channel_counts = [] # [512,256,128,64,32] [1/16,1/8,1/4,1/2,1]

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        
        for i in range(self.network_depth):
            if i == 0:
                self.encodingBlocks.append(encoder_block_yb_v1(img_ch, features, reduction_ratio=self.reduction_ratio, att_mode = att_mode, conv_type = conv_type, first_block = True))
                decoder_channel_counts.insert(0, features)
            else:
                self.encodingBlocks.append(encoder_block_yb_v1(features, 2*features, reduction_ratio=self.reduction_ratio, att_mode = att_mode, conv_type = conv_type))
                decoder_channel_counts.insert(0, 2*features)
                features *= 2
        
        self.decodingBlocks = nn.ModuleList([])
        self.headBlocks = nn.ModuleList([])
        self.convBlocks = nn.ModuleList([])
        for i in range(self.network_depth-1):
            self.decodingBlocks.append(decoder_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1]))
            # multi heads
            if i >= self.network_depth-1-self.n_head:
                self.headBlocks.append(head_block(decoder_channel_counts[i], decoder_channel_counts[-1], 2**(self.network_depth-1-i)))
        for i in range(self.n_skip):
            if conv_type == 'basic':
                self.convBlocks.append(conv_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], reduction_ratio=self.reduction_ratio, att_mode = att_mode))
            elif conv_type == 'sk':
                self.convBlocks.append(skconv_block_yb_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], is_shortcut = is_shortcut))
                
        self.activation = activation
        self.seg_head = nn.Conv2d(init_features, n_classes, 1)
        
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
                                nn.Linear(decoder_channel_counts[0], 256),
                                nn.ReLU(),
                                nn.Linear(256, 32),
                                nn.ReLU(),
                                nn.Linear(32, self.n_classes)
                                )
        
    def forward(self, x):
        skip_connections = [] # [1,1/2,1/4,1/8]
        for i in range(self.network_depth):
            x = self.encodingBlocks[i](x)
            if i<self.network_depth-1:
                skip_connections.append(x)
                
        cls_output = self.AvgPool(x)
        cls_output = cls_output.view(cls_output.size()[0], -1) # flatten
        cls_output = self.activation(self.cls_head(cls_output))
        
        x_heads = []
        for i in range(self.network_depth-1):
            if i>=self.network_depth-1-self.n_head:
                x_head = x
                x_heads.append(self.headBlocks[i - self.network_depth+1+self.n_head](x_head))
            
            x = self.decodingBlocks[i](x)
            if self.n_skip>0:
                self.n_skip = self.n_skip-1
                skip = skip_connections.pop()
                x = self.convBlocks[i](torch.cat([x, skip], 1))
            
        n_head = len(x_heads)
        for i, x_head in enumerate(x_heads):
            if i==0:
                x_seg=self.seg_head(x_head)
            else:
                x_seg+=self.seg_head(x_head)
                
        seg_output = self.activation(x_seg/n_head) 
#         return seg_output, cls_output
        return seg_output
    
class sk(nn.Module):
    '''reduction_ratio = None -> U-NET
    '''

    def __init__(self, img_ch=3, n_classes=2, init_features=32, network_depth=5, reduction_ratio=8, n_skip=4, att_mode = None, is_shortcut = False, conv_type = 'sk', activation = nn.LogSoftmax(dim=1)):
        super(sk, self).__init__()
        self.n_classes = n_classes
        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.n_skip = n_skip # [1/16,1/8,1/4,1/2,1] [0,1,2,3,4], must <=self.network_depth-1
        
        decoder_channel_counts = [] # [512,256,128,64,32] [1/16,1/8,1/4,1/2,1]

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        
        for i in range(self.network_depth):
            if i == 0:
                self.encodingBlocks.append(encoder_block_yb_v1(img_ch, features, reduction_ratio=self.reduction_ratio, att_mode = att_mode, conv_type = conv_type, first_block = True))
                decoder_channel_counts.insert(0, features)
            else:
                self.encodingBlocks.append(encoder_block_yb_v1(features, 2*features, reduction_ratio=self.reduction_ratio, att_mode = att_mode, conv_type = conv_type))
                decoder_channel_counts.insert(0, 2*features)
                features *= 2
        
        self.decodingBlocks = nn.ModuleList([])
        self.convBlocks = nn.ModuleList([])
        for i in range(self.network_depth-1):
            self.decodingBlocks.append(decoder_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1]))
        for i in range(self.n_skip):
            if conv_type == 'basic':
                self.convBlocks.append(conv_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], reduction_ratio=self.reduction_ratio, att_mode = att_mode))
            elif conv_type == 'sk':
                self.convBlocks.append(skconv_block_yb_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], is_shortcut = is_shortcut))
                
        self.activation = activation
        self.seg_head = nn.Conv2d(init_features, n_classes, 1)
        
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
                                nn.Linear(decoder_channel_counts[0], 256),
                                nn.ReLU(),
                                nn.Linear(256, 32),
                                nn.ReLU(),
                                nn.Linear(32, self.n_classes)
                                )
        
    def forward(self, x):
        skip_connections = [] # [1,1/2,1/4,1/8]
        for i in range(self.network_depth):
            x = self.encodingBlocks[i](x)
            if i<self.network_depth-1:
                skip_connections.append(x)
                
        cls_output = self.AvgPool(x)
        cls_output = cls_output.view(cls_output.size()[0], -1) # flatten
        cls_output = self.activation(self.cls_head(cls_output))
            
        for i in range(self.network_depth-1):
            x = self.decodingBlocks[i](x)
            if self.n_skip>0:
                self.n_skip = self.n_skip-1
                skip = skip_connections.pop()
                x = self.convBlocks[i](torch.cat([x, skip], 1))

        # seg_output = self.activation(self.seg_head(x)) 
        smx = torch.nn.Softmax2d()
        seg_output = smx(self.seg_head(x))
#         return seg_output, cls_output
        return seg_output
    
class UNet(nn.Module):
    '''reduction_ratio = None -> U-NET
    '''

    def __init__(self, img_ch=3, n_classes=2, init_features=32, network_depth=5, reduction_ratio=8, n_skip=4, att_mode = 'cbam', activation = nn.LogSoftmax(dim=1)):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.n_skip = n_skip # [1/16,1/8,1/4,1/2,1] [0,1,2,3,4], must <=self.network_depth-1
        
        decoder_channel_counts = [] # [512,256,128,64,32] [1/16,1/8,1/4,1/2,1]

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            if i == 0:
                self.encodingBlocks.append(encoder_block_yb(img_ch, features, reduction_ratio=self.reduction_ratio, att_mode = att_mode, first_block = True))
                decoder_channel_counts.insert(0, features)
            else:
                self.encodingBlocks.append(encoder_block_yb(features, 2*features, reduction_ratio=self.reduction_ratio, att_mode = att_mode))
                decoder_channel_counts.insert(0, 2*features)
                features *= 2
        
        self.decodingBlocks = nn.ModuleList([])
        self.convBlocks = nn.ModuleList([])
        for i in range(self.network_depth-1):
            self.decodingBlocks.append(decoder_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1]))
        for i in range(self.n_skip):
            self.convBlocks.append(conv_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], reduction_ratio=self.reduction_ratio, att_mode = att_mode))
        self.activation = activation
        self.seg_head = nn.Conv2d(init_features, n_classes, 1)
        
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
                                nn.Linear(decoder_channel_counts[0], 256),
                                nn.ReLU(),
                                nn.Linear(256, 32),
                                nn.ReLU(),
                                nn.Linear(32, self.n_classes)
                                )
        
    def forward(self, x):
        skip_connections = [] # [1,1/2,1/4,1/8]
        for i in range(self.network_depth):
            x = self.encodingBlocks[i](x)
            if i<self.network_depth-1:
                skip_connections.append(x)
                
        cls_output = self.AvgPool(x)
        cls_output = cls_output.view(cls_output.size()[0], -1) # flatten
        cls_output = self.activation(self.cls_head(cls_output))
            
        for i in range(self.network_depth-1):
            x = self.decodingBlocks[i](x)
            if self.n_skip>0:
                self.n_skip = self.n_skip-1
                skip = skip_connections.pop()
                x = self.convBlocks[i](torch.cat([x, skip], 1))
                
#         for i in range(self.network_depth-1):
#             x = self.decodingBlocks[i](x)
#             skip = skip_connections.pop()
#             x = self.convBlocks[i](torch.cat([x, skip], 1))

        seg_output = self.activation(self.seg_head(x)) 
        return seg_output, cls_output

def test():
    net = UNet_V2(network_depth=5, n_head = 2)
#     net = SpatialGate()
    print(net)
    y, _ = net(torch.randn(1,3,400,400))
    print(y.size())
    
if __name__ == '__main__':
    test()
