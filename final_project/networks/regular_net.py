import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNN(nn.Module):
    def __init__(self, num_hidden_layers: int = 1) -> None:
        """Initialize CNN.

        Args:
            num_hidden_layers: number of hidden layers
        """
        super().__init__()
        layers = [nn.Conv2d(1, 32, kernel_size=3, padding=0),
                  nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=0))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).reshape(-1, 9).squeeze() # return as [[1,2,3,4,5,6,7,8,9]]