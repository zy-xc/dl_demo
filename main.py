import os
import shutil
import argparse
import time

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

from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

# from vgg16 import VGG16
# from transform_net import TransformerNet
import models
from config import DefaultConfig
import utils


def train(**kwargs):
    config = DefaultConfig()
    config.parse({
        'batch_size' : 8,
    })
    config.parse(kwargs)

    # ================================================================== #
    # 1. device                                                          #
    # ================================================================== #
    device_ids = [0, 1, 2, 3]
    device = torch.device('cuda', device_ids[0])

    # ================================================================== #
    # 2. model and optimizer                                             #
    # ================================================================== #
    # model
    model = models.BasicModule()
    # optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=config.lr, weight_decay=config.weight_decay)
    
    # ================================================================== #
    # 2. dataset and data_loader                                         #
    # ================================================================== #
    train_set = torchvision.datasets.ImageFolder(
        root=config.train_data_folder,
        transform=utils.transform,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # ================================================================== #
    # 3. train model                                                     #
    # ================================================================== #
    steps_per_epoch = len(train_loader)
    for epoch in range(config.num_epochs):
        with tqdm(enumerate(train_loader), total=steps_per_epoch) as pbar:
            for i, (images, labels) in pbar:
                if images.size(0) != config.batch_size:
                    continue

                # 1. to gpu
                images = images.to(device, non_blocking=config.non_blocking)
                labels = labels.to(device, non_blocking=config.non_blocking)

                # 2. forward
                loss = 0.0

                # 3. backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # output loss
                if (i+1) % config.print_freq == 0:
                    print('epoch = {}, i+1 = {}, loss = {}'
                          .format(epoch+1, i+1, loss.item()))

        # save model
        model_save_name = 'epoch_{}.pth'.format(epoch+1)
        model_save_path = os.path.join(config.checkpoints_folder, model_save_name)
        model.save(model_save_path)


if __name__ == '__main__':
    config = DefaultConfig()
    # config.parse({
    #     'style_img_path' : os.path.join(config.base_dir, 'images/style/candy.jpg'),
    # })
    train()

