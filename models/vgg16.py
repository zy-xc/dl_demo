import os
import shutil

# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


__all__ = ['VGG16']


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_features = torchvision.models.vgg16(pretrained=True).features
        self.features = vgg_features[:23]
        self.perceptual_layers = {3, 8, 15, 22}
        # self.slice1 = nn.Sequential()
        # self.slice2 = nn.Sequential()
        # self.slice3 = nn.Sequential()
        # self.slice4 = nn.Sequential()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.perceptual_layers:
                out.append(x)
        return out


if __name__ == '__main__':
    pass


