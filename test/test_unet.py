import os
import sys
import warnings

import torchsummary
import torch
import numpy as np

sys.path.append(os.getcwd() +'/src')
from models.unet_model import UNet,Stem,DownSample,UpSample,OutConv


def test_stem():
    block = Stem(3,64)
    x = torch.randn(1,3,256,256)
    y = block(x)

    assert y.shape == torch.Size([1,64,256,256])


def test_downblock():
    block = DownSample(64,128)
    x = torch.randn(1,64,256,256)
    y = block(x)

    assert y.shape == torch.Size([1,128,128,128])

def test_upblock():
    block = UpSample(128,64)
    x = torch.randn(1,128,128,128)
    x_skip = torch.randn(1,64,256,256)
    y = block(x,x_skip)

    assert y.shape == torch.Size([1,64,256,256])

def test_outblock():
    block = OutConv(64,2)
    x = torch.randn(1,64,256,256)
    y = block(x)

    assert y.shape == torch.Size([1,2,256,256])

def test_unet():
    model = UNet(2,3)
    x = torch.randn(1,3,256,256)
    y = model(x)

    assert y.shape == torch.Size([1,2,256,256])
    
