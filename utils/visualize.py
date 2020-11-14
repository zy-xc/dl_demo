import os, shutil

from torch.utils.tensorboard import SummaryWriter


class Visualize:
    def __init__(self, log_dir, remove_pre=True):
        if remove_pre:
            shutil.rmtree(log_dir, ignore_errors=True)
        self.writer = SummaryWriter(log_dir)
    
    def __getattr__(self, name):
        return getattr(self.writer, name)


