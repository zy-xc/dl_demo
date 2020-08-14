import os
import shutil

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 防止报错: image file is truncated 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class RangeResizedCrop(object):
    """自定义图片转换函数
    resize image to kkep the smallest dimension in the range(eg: [256, 480]),
    and randomly cropped regions of size(eg:(256,256))
    Args:
        size (int or sequence) – expected output size of each edge.
        range (sequence): [min, max]
    """

    def __init__(self, size=256, range=(256,480)):
        if not isinstance(size, (int, tuple, list)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if len(range) != 2:
            raise ValueError("Please provide only two dimensions (min, max) for range.")
        self.size = size
        self.range = range

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        low, high = self.range[0], self.range[1]
        ratio = 1
        if low > high:
            low, high = high, low
        width, height = img.size
        if width < low or height < low:
            ratio = low / min(width, height)
        elif width > high and height > high:
            ratio = high / min(width, height)
        width, height = int(width*ratio), int(height*ratio)
        img = img.resize((width, height))
        img = transforms.RandomCrop(self.size, pad_if_needed=True)(img)
        return img


dataset_mean_tensor = torch.tensor([0.485, 0.456, 0.406])
dataset_std_tensor = torch.tensor([0.229, 0.224, 0.225])

transform = transforms.Compose([
    RangeResizedCrop(size=(256,256), range=(256,480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean_tensor, std=dataset_std_tensor),
])

de_norm = transforms.Compose([
    transforms.Normalize(mean=(-dataset_mean_tensor/dataset_std_tensor),
                         std=(1/dataset_std_tensor)),
    # transforms.ToPILImage(),
])


def load_image(image_path, device='cpu', unsqueeze=True):
    if type(device) == str:
        device = torch.device(device)
    elif type(device) == int:
        device = torch.device('cuda', device)
    elif type(device) != torch.device:
        raise ValueError('type of argument \'device\' should be str, int, or torch.device, got {}'
                         .format(type(device)))
    img = Image.open(image_path)
    # img = img.resize(imsize)
    tensor = transform(img)
    if unsqueeze:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor):
    """恢复单张图片，输入shape须为(1,C,H,W)或(C,H,W)"""
    tensor = tensor.detach().clone().cpu()
    if len(tensor.shape) == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    elif len(tensor.shape) == 4:
        raise ValueError('tensor shape should be 1xCxHxW or CxHxW, got {}'.format(tensor.shape))
    
    image = de_norm(tensor)
    image.clamp_(0, 1)
    image = transforms.ToPILImage()(image)
    
    return image

def recover_batch_tensor_to_batch_images(tensor: torch.Tensor):
    """从斯威tensor中恢复图片, 输入shape须为(N,C,H,W)或(C,H,W),
       若为(C,H,W), 则unsqueeze为(1,C,H,W),
       返回shape为: (N, C, H, W)
    """
    tensor = tensor.detach().clone().cpu()
    if len(tensor.size()) == 3:
        tensor = tensor.unsqueeze(0)
    if len(tensor.size()) != 4:
        raise ValueError('tensor shape should be NxCxHxW or CxHxW, got {}'.format(tensor.shape))
    tensor = tensor * dataset_std_tensor.view(1,3,1,1) + dataset_mean_tensor.view(1,3,1,1)
    tensor = tensor.clamp(0,1).cpu()
    return tensor

def plt_show_image(tensor: torch.Tensor):
    # tensor = tensor.cpu().detach().clone()
    # tensor = tensor.squeeze(0)
    # img = de_norm(tensor)
    # img.clamp_(0, 1)
    # img = transforms.ToPILImage()(img)
    if len(tensor.shape) != 3 and len(tensor.shape) != 4:
        raise ValueError('tensor shape should be NxCxHxW or CxHxW, got {}'.format(tensor.shape))
    if len(tensor.shape) == 3:
        image = tensor_to_image(tensor)
        np_image = np.array(image)
    elif len(tensor.shape) == 4:
        tensor = recover_batch_tensor_to_batch_images(tensor)
        image = torchvision.utils.make_grid(tensor, nrow=4)
        np_image = image.numpy()
        np_image = np.transpose(np_image, (1, 2, 0))
    plt.imshow(np_image)

def gram_matrix(X: torch.Tensor):
    N, C, H, W = X.size()
    X = X.view(N, C, H * W)
    gram = X.bmm(X.transpose(1, 2))
    return gram / (C * H * W)
    

if __name__ == '__main__':
    pass