import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm2d

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        # input image 600 x 800
        # (n, 3, 600, 800)
        self.layer1 = nn.Sequential(

        )

    def forward(self, x):
        out = None

        return out
