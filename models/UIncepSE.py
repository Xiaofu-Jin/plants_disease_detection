from .UNet import *
from .SENet import *
import torchvision
import torch
import torch.nn.functional as F
from torch import nn

class ConcatLayer(nn.Module):
    def __init__(self, unet):
        super(ConcatLayer, self).__init()
        self.conv = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3)
        )
        self.unet = self.conv(unet)
    def forward(self, x):
        u = torch.mul(x, self.unet)
        result = torch.cat(x, u, dim=1)
        final = BasicConv2d(384, 192, kernel_size=1)
        return final
        
 
class DoubleAttention(nn.Module):
    def __init__(self, num_classes, unet, aux_logits=True, transform_input=False):       
        super(DoubleAttention, self).__init__()
        model = Inception3(num_classes=num_classes, aux_logits=aux_logits,
                           transform_input=transform_input)
        model.Conv2d_4a_3x3.add_module("ConcatLayer", ConcatLayer(unet))
        model.Mixed_5b.add_module("SELayer", SELayer(192))
        model.Mixed_5c.add_module("SELayer", SELayer(256))
        model.Mixed_5d.add_module("SELayer", SELayer(288))
        model.Mixed_6a.add_module("SELayer", SELayer(288))
        model.Mixed_6b.add_module("SELayer", SELayer(768))
        model.Mixed_6c.add_module("SELayer", SELayer(768))
        model.Mixed_6d.add_module("SELayer", SELayer(768))
        model.Mixed_6e.add_module("SELayer", SELayer(768))
        if aux_logits:
            model.AuxLogits.add_module("SELayer", SELayer(768))
        model.Mixed_7a.add_module("SELayer", SELayer(768))
        model.Mixed_7b.add_module("SELayer", SELayer(1280))
        model.Mixed_7c.add_module("SELayer", SELayer(2048))
                                                                             
        self.model = model
    def forward(self, x):
        _, _, h, w = x.size()
        if (h, w) != (299, 299):
            raise ValueError("input size must be (299, 299)")
                                                          
        return self.model(x)
