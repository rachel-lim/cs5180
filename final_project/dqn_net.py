import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self):
        super().__init__()
        # self.n_inv = 3 * n_theta * n_p
        self.conv = torch.nn.Sequential(
            # # 128x128
            # nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # # 64x64
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # # 32x32
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            
            # 15x15
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 13x13
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 11x11
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 9x9
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 7x7
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 5x5
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        # self.n_p = n_p
        # self.n_theta = n_theta

        # for m in self.named_modules():
        #     if isinstance(m[1], nn.Conv2d):
        #         # nn.init.kaiming_normal_(m[1].weight.data)
        #         nn.init.xavier_normal_(m[1].weight.data)
        #     elif isinstance(m[1], nn.BatchNorm2d):
        #         m[1].weight.data.fill_(1)
        #         m[1].bias.data.zero_()

    def forward(self, x):
        # batch_size = x.shape[0]
        q = self.conv(x)
        # q = q.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return q