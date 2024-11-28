import torch

from e2cnn import gspaces
from e2cnn import nn

# 3x3 out
class EquivariantCNNCom(torch.nn.Module):
    def __init__(self, initialize=True):
        super().__init__()
        # self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 7x7
            nn.R2Conv(nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            # 5x5
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),  # 1 channel
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))  # convert to gtensor
        out = self.conv(x).tensor.reshape(-1, 9).squeeze()  # convert back to tensor and squeeze

        return out