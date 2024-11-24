import torch

from e2cnn import gspaces
from e2cnn import nn

# 3x3 out
class EquivariantCNNCom(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        # self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # # 128x128
            # nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
            #           nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
            #           kernel_size=3, padding=1, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), 2),
            # # 64x64
            # nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            #           nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           kernel_size=3, padding=1, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # # 32x32
            # nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            #           nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
            #           kernel_size=3, padding=1, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            
            # 15x15 trivial to regular
            nn.R2Conv(nn.FieldType(self.r2_act, 1*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            # 15x15
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            # 13x13
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            # 11x11
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            # 9x9
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 7x7
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            # 5x5
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]),  # 1 channel
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        out = self.conv(x).tensor #.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return out