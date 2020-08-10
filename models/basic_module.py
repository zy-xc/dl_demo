import os
import warnings

import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self)) # 默认名字
    
    def save(self, save_path):
        save_folder = os.path.dirname(save_path)
        assert os.path.isdir(save_folder), \
               '{} is not a directory'.format(save_folder)
        
        _, extension = os.path.splitext(save_path)
        if not extension.endswith('pth'):
            warnings.warn('expect ext as \'.pth\', but got {}'.format(extension))

        torch.save(self.state_dict(), save_path)
    
    def load(self, model_path, map_location=None):
        self.load_state_dict(torch.load(model_path, map_location=map_location))