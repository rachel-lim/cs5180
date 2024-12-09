import torch

from e2cnn import gspaces
from e2cnn import nn

# 3x3 out
class EquiCNN(torch.nn.Module):
    def __init__(self, num_hidden_layers: int = 1, initialize: bool = True) -> None:
        """Initialize rotational-equivariant CNN"""
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(4)

        layers = [nn.R2Conv(nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),
                  nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                  kernel_size=3, padding=0, initialize=initialize),
                  nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True)]
        for _ in range(num_hidden_layers):
            layers.append(nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                                    nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                                    kernel_size=3, padding=0))
            layers.append(nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True))
        layers.append(nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]),  # 1 channel
                      kernel_size=3, padding=1))
        
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))  # convert to gtensor
        out = self.layers(x).tensor.reshape(-1, 9).squeeze()  # convert back to tensor and squeeze

        return out