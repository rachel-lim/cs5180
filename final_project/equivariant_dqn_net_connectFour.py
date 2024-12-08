import torch

from e2cnn import gspaces
from e2cnn import nn

class EquivariantCNNCom(torch.nn.Module):
    def __init__(self, initialize=True):
        super().__init__()
        self.r2_act = gspaces.Flip2dOnR2()  # reflection about y axis
        self.conv = torch.nn.Sequential(
            # 7x7
            nn.R2Conv(nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # 7x7
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # 7x7
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),  # 1 channel
                      kernel_size=3, padding=1),
            # 7x7
            # nn.R2Conv(nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),
            #           nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           kernel_size=3, padding=0, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # # 7x7
            # nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           kernel_size=3, padding=0),
            # nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           kernel_size=3, padding=0),
            # nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # # 7x7
            # nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),  # 1 channel
            #           kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))  # convert to gtensor
        out = self.conv(x).tensor.sum(axis=2).squeeze()  # convert back to tensor and squeeze
        # out = self.conv(x).tensor.reshape(-1, 9).squeeze()
        return out