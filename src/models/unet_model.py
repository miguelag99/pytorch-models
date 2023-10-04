import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    ## Class for Unet Model
    def __init__(self,n_class, n_channels=3, bilinear=False):
        super().__init__()

        self.n_class = n_class
        self.n_channels = n_channels
        
        # Encoder layers
        self.in_conv = Stem(n_channels,64)
        self.down1 = DownSample(64,128)
        self.down2 = DownSample(128,256)
        self.down3 = DownSample(256,512)
        self.down4 = DownSample(512,1024)

        # Decoder layers
        self.up1 = UpSample(1024,512,bilinear)
        self.up2 = UpSample(512,256,bilinear)
        self.up3 = UpSample(256,128,bilinear)
        self.up4 = UpSample(128,64,bilinear)
        self.out_conv = OutConv(64,n_class)

    def forward(self,x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.out_conv(x)

        return x




class Stem(nn.Module):
    ## Input Convolution Block for UNet 2*(Conv+BN+ReLU)
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    
    def forward(self,x):
        return self.conv(x)
    
class OutConv(nn.Module):
    ## Output Convolution Block for UNet (Conv)
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,kernel_size=1)

    
    def forward(self,x):
        return self.conv(x)

class DownSample(nn.Module):
    ## Downsample block for UNet (Pool + 2x(conv+BN+ReLU))) 

    def __init__(self,in_c,out_c):
        super().__init__()

        self.conv_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.conv_pool(x)
    
class UpSample(nn.Module):
    ## Upsample block for UNet (Upsample + 2x(conv+BN+ReLU)))

    def __init__(self,in_c,out_c,bilinear=True):
        super().__init__()
        if bilinear == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c,in_c//2,kernel_size=1)) # Added to fix channel coherence
            self.conv = nn.Sequential(
                nn.Conv2d(in_c,in_c // 2,kernel_size=3,padding=1),
                nn.BatchNorm2d(in_c // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c // 2,out_c,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_c,in_c // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )


    
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)


