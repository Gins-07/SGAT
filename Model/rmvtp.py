import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# attention module
# from Model.bam import BAM
# from cbam import CBAM
# # from Model.se import SELayer
# from  sk import SKConv

from vitrlp import *
from einops import rearrange
'this is based on the original resnet'

class basic_conv (nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv =nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=7,stride=2,padding=3,bias=False),
                                 nn.BatchNorm2d(ch_out),
                                 nn.ReLU(inplace=True)
        )
    
    def forward (self,x):
        out = self.conv(x)
        return out
    
class BottleBlock(nn.Module):
    
    expansion =2 # the multiply factor in channel
    
    def __init__(self,in_channel,mid_channel,stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, mid_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.batch1 = nn.BatchNorm2d(mid_channel)
        self.relu1 =nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batch2 = nn.BatchNorm2d(mid_channel)
        self.relu2 =nn.ReLU(inplace = True)
        self.conv3 = nn.Conv2d(mid_channel, mid_channel*self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.batch3 = nn.BatchNorm2d(mid_channel*self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride !=1 or in_channel != mid_channel*self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel,mid_channel*self.expansion,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(mid_channel*self.expansion))
        
        self.relu = nn.ReLU(inplace = True)
    def forward(self,x):
        
        # fed into the first convolution
        output = self.conv1(x)
        output = self.batch1(output)
        output = self.relu1(output)
        
        # fed into the second convolution
        output = self.conv2(output)
        output = self.batch2(output)
        output = self.relu2(output)
        
        # fed into the third convolution
        output = self.conv3(output)
        output = self.batch3(output)
        
        # the residual connection
        
        output += self.shortcut(x)
        output = self.relu(output)
        
        return output
    
class BottleBlock_vt(nn.Module):
    
    expansion =2 # the multiply factor in channel
    
    def __init__(self,in_channel=32,mid_channel=32,stride=1,img_size=128,patch_size_vit=4,embed_dim_vit=96,depth_vit=4,num_heads_vit=4):
        super().__init__()
        
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channel, mid_channel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channel, mid_channel*self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        )
        
        self.batch3 = nn.BatchNorm2d(mid_channel*self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride !=1 or in_channel != mid_channel*self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel,mid_channel*self.expansion,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(mid_channel*self.expansion))
        
        self.relu = nn.ReLU(inplace = True)
        
        num_vit = (img_size//stride//patch_size_vit)*(img_size//stride//patch_size_vit)
        #embed layer
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size_vit, in_c=mid_channel*self.expansion, embed_dim=embed_dim_vit)
        self.pos_drop = nn.Dropout(p=0.)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_vit, embed_dim_vit))
        self.norm =nn.LayerNorm(embed_dim_vit)
        
        # transformer layer
        self.embed_dim = embed_dim_vit
        self.patch_size = patch_size_vit
        
        self.vit1 = VisionTransformer(img_size=img_size,
                              patch_size=patch_size_vit,
                              embed_dim=embed_dim_vit,
                              depth=depth_vit,
                              num_heads=num_heads_vit,
                              in_c = in_channel)
        
        num_vit = (img_size//stride//patch_size_vit)*(img_size//stride//patch_size_vit)
        
        self.fcs = nn.Linear(embed_dim_vit,embed_dim_vit//2)
        self.normcs = nn.LayerNorm(embed_dim_vit//2)
        self.fcs1 =  nn.Linear(embed_dim_vit//2, embed_dim_vit)
        self.normcs1 =nn.LayerNorm(embed_dim_vit)
        self.fcs2 =  nn.Linear(embed_dim_vit//2,embed_dim_vit)
        self.normcs2 = nn.LayerNorm(embed_dim_vit)
        
        self.fcs_final = nn.Linear(embed_dim_vit,patch_size_vit*patch_size_vit*embed_dim_vit)
        self.norm_final = nn.LayerNorm(embed_dim_vit)
        
        self.conv = conv_block_yb(embed_dim_vit,mid_channel*self.expansion,reduction_ratio=None, att_mode = None)
        # self.conv = nn.Conv2d(embed_dim_vit,mid_channel*self.expansion,kernel_size=1,stride=1,bias=False)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(x) 
        return x
    
    def forward(self,x):
        B,C,H,W = x.shape

        resoutput = self.res(x)
        resoutput = self.batch3(resoutput)
        
        # the residual connection
        
        resoutput += self.shortcut(x)
        resoutput = self.relu(resoutput)
        #embed the output
        res = self.forward_features(resoutput) #b,n(n1+1),embed_dim
        res = res.unsqueeze(dim=1)
        vit = self.vit1(x)
        vit = vit.unsqueeze(dim=1)
        
        fea = torch.cat([res,vit],dim=1) #b,2,n,embed_dim
        fea_sum = torch.sum(fea,dim=1) #b,n,embed_dim
        fea_mean = fea_sum.mean(dim=1)
        
        fea_combine_sum_z = self.fcs(fea_mean)#b,dim/2
        fea_combine_sum_z = self.normcs(fea_combine_sum_z)
        
        fea_res =self.fcs1(fea_combine_sum_z).unsqueeze(dim=1)#b,1,dim
        fea_res = self.normcs1(fea_res)
        fea_vit =self.fcs2(fea_combine_sum_z).unsqueeze(dim=1)#b,1,dim
        fea_vit = self.normcs2(fea_vit)
        
        attention_fea = torch.cat((fea_res,fea_vit),dim=1)#b,2,dim
        attention_factors = self.softmax(attention_fea)#b,2,dim
        attention_factors = attention_factors.unsqueeze(dim=2)#b,2,1,dim
        
        fea = (fea *attention_factors).sum(dim=1)#b,n,dim
        
        attention_fea = self.fcs_final(fea)#b,n,dim*patch_size*patch_size
        attention_fea = attention_fea.view(B, H//self.patch_size, W//self.patch_size, -1)
        attention_fea = rearrange(attention_fea, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size, c=self.embed_dim)
        attention_fea = attention_fea.view(B,-1,self.embed_dim)
        attention_fea = self.norm_final(attention_fea)
        
        attention_fea = attention_fea.view(B,H,W,-1).permute(0,3,1,2)
        
        x_attention = self.conv(attention_fea)

        return x_attention    
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

        return out 

class resmvtp(nn.Module):
    '''reduction_ratio = None -> U-NET
    '''

    def __init__(self, img_ch=3, n_classes=2):
        super().__init__()
        init_features=32
        network_depth=5
        reduction_ratio=8
        n_skip=4
        att_mode = None
        is_shortcut = False
        self.n_classes = n_classes
        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.n_skip = n_skip # [1/16,1/8,1/4,1/2,1] [0,1,2,3,4], must <=self.network_depth-1
        layer = [3,4,6,3]
        self.in_planes = 32
        decoder_channel_counts = [] # [512,256,128,64,32] [1/16,1/8,1/4,1/2,1]
        
        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        
        res_features =16
            
        for i in range(self.network_depth):
            if i == 0:
                self.encodingBlocks.append(basic_conv(img_ch, self.in_planes))
                decoder_channel_counts.insert(0, features)
                self.encodingBlocks.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
                decoder_channel_counts.insert(0, features)
            elif i==1:
                self.encodingBlocks.append(self._make_layer(BottleBlock_vt,features,layer[0]))
                decoder_channel_counts.insert(0, 2*features)
                features *= 2
            else:
                self.encodingBlocks.append(self._make_layer(BottleBlock,features,layer[i-1],stride=2))
                decoder_channel_counts.insert(0, 2*features)
                features *= 2
                    
        
        self.decodingBlocks = nn.ModuleList([])
        self.convBlocks = nn.ModuleList([])
            
        for i in range(self.network_depth-1):

            self.decodingBlocks.append(decoder_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1]))
    
        for i in range(self.n_skip):

            self.convBlocks.append(conv_block_yb(decoder_channel_counts[i], decoder_channel_counts[i+1], reduction_ratio=self.reduction_ratio, att_mode = att_mode))
            
        self.up2 = decoder_block_yb(init_features,init_features)
        self.seg_head = nn.Conv2d(init_features, n_classes, 1)
        
        for m in self.modules():
            
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        for m in self.modules():
            if isinstance(m,BottleBlock):
                nn.init.constant_(m.batch3.weight,0)
            elif isinstance(m,BottleBlock_vt):
                nn.init.constant_(m.batch3.weight,0)
    
    def _make_layer(self,block,in_channel,num_layers,stride=1):
        
        layers =[]
        layers.append(block(self.in_planes,in_channel,stride))
        self.in_planes = in_channel*block.expansion
        for i in range(1,num_layers):
            layers.append(block(self.in_planes,in_channel))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        skip_connections = [] # [1,1/2,1/4,1/8]
        for i in range(self.network_depth+1):
            x = self.encodingBlocks[i](x)
            if i!=1 and i<self.network_depth:
                skip_connections.append(x)
                
        for i in range(self.network_depth-1):
            x = self.decodingBlocks[i](x)
            if  self.n_skip>0:
                self.n_skip = self.n_skip-1
                skip = skip_connections.pop()
                x = self.convBlocks[i](torch.cat([x, skip], 1))
          
        seg_output = self.up2(x)
        seg_output = self.seg_head(seg_output)
    
        return seg_output
    
