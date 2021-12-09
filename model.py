import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm2d

class ConvNN(nn.Module):

    # I have no idea what I'm doing
    def __init__(self, num_classes: int = 5, dropout: float=0.5):
        super(ConvNN, self).__init__()
        # input image 600 x 800
        # (n, 3, 600, 800)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 300 x 400
        ) # 256
        # (n, 25, 300, 400)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 150 x 200
        ) # 128

        # (n, 50, 150, 200)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 75 x 100
        ) # 64

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 37 x 50
        ) # 32

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 18 x 25
        ) # 16

        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 9 x 12
        ) # 8

        # (n, 50, 9, 12)
        self.hidden = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x: torch.Tensor):

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out6 = torch.flatten(out6, 1)
        out = self.hidden(out6)
        return out