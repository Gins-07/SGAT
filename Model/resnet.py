import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
        
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.apply(_init_vit_weights)

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
        self.apply(_init_vit_weights)
        
    def forward(self,x):
        x = self.up(x)
        return x

class BasicBlock(nn.Module):
    
    expansion =1 # the multiply factor in channel
    
    def __init__(self,in_channel,out_channel,stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batch1 = nn.BatchNorm2d(out_channel)
        self.relu1 =nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch2 = nn.BatchNorm2d(out_channel)
        
        self.shortcut = nn.Sequential()
        if stride !=1 or in_channel != out_channel*self.expansion:
            self.shortcut =nn.Sequential(
                nn.Conv2d(in_channel,out_channel*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )
        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self,x):
        
        # fed into the first convolution
        output = self.conv1(x)
        output = self.batch1(output)
        output = self.relu1(output)
        
        # fed into the second convolution
        output = self.conv2(output)
        output = self.batch2(output)
        
        # the residual connection
        
        output += self.shortcut(x)
        output = self.relu(output)
        
        return output

class BottleBlock(nn.Module):
    
    expansion =4 # the multiply factor in channel
    
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
    
class ResNet(nn.Module):
    
    def __init__(self,in_channel,out_channel):
        super().__init__()
        
        self.in_planes =64 #define the inital dimension of channel
        block = BottleBlock
        layer = [3,4,6,3]
        #decrease the size of image to 1/4
        self.conv1 = nn.Conv2d(in_channel,self.in_planes,kernel_size=7,stride=2,padding=3,bias=False)
        self.batch1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # the four blocks in resnet
        self.layer1 = self._make_layer(block,64,layer[0])
        self.layer2 = self._make_layer(block,128,layer[1],stride=2)
        self.layer3 = self._make_layer(block,256,layer[2],stride=2)
        self.layer4 = self._make_layer(block,512,layer[3],stride=2)
        
        # the decoder
        self.Up5 = up_conv(ch_in=2048,ch_out=1024)
        self.Up_conv5 = conv_block(ch_in=2048, ch_out=1024)

        self.Up4 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv4 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up3 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv3 = conv_block(ch_in=512, ch_out=256)
        
        self.Up2 = up_conv(ch_in=256,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=2*64, ch_out=64)
        
        self.Up1 = up_conv(ch_in=64, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64,out_channel,kernel_size=1)
        
        for m in self.modules():
            
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        for m in self.modules():
            if isinstance(m,BottleBlock):
                nn.init.constant_(m.batch3.weight,0)
            elif isinstance(m,BasicBlock):
                nn.init.constant_(m.batch2.weight,0)
    
    def _make_layer(self,block,in_channel,num_layers,stride=1):
        
        layers =[]
        layers.append(block(self.in_planes,in_channel,stride))
        self.in_planes = in_channel*block.expansion
        for i in range(1,num_layers):
            layers.append(block(self.in_planes,in_channel))

        return nn.Sequential(*layers)
        
        
    
    def forward(self,x):
        
        x1 = self.relu1(self.batch1(self.conv1(x)))
        x2 = self.maxpool(x1)
        
        x2 = self.layer1(x2)
        
        x3 = self.layer2(x2)

        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d2 =self.Up1(d2)

        d1 = self.Conv_1x1(d2)
    

        return d1
        
        
        