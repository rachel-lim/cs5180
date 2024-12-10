import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 7x7
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 7x7
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x).sum(axis=2).squeeze() # sum columns before returning
        # return self.layers(x).reshape(-1, 9).squeeze()