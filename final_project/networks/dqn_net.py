import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(          
            # 7x7
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 5x5
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 3x3
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x).reshape(-1, 9).squeeze() # return as [[[1,2,3,4,5,6,7,8,9]]]