import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(512, 18, kernel_size=1, padding=0),
        )

        self.n_p = n_p
        self.n_theta = n_theta

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.conv(x)
        q = q.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return q