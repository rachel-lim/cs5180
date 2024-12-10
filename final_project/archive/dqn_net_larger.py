import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(          
            # # 15x15
            # nn.Conv2d(1, 32, kernel_size=3, padding=0),
            # nn.ReLU(inplace=True),
            # 13x13
            # nn.Conv2d(1, 64, kernel_size=3, padding=0),
            # nn.ReLU(inplace=True),
            # # 11x11
            # nn.Conv2d(64, 64, kernel_size=3, padding=0),
            # nn.ReLU(inplace=True),
            # 9x9
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 5x5
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 3x3
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x = self.layers(x).squeeze()
        # out = self.out_layer(x)

        # return out
        return self.layers(x).reshape(-1, 9).squeeze() # return as [[[1,2,3,4,5,6,7,8,9]]]

    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (2, 9),
            'kwargs': {
                'num_layers': 3,
                'hidden_dim': 1,
            },
            'state_dict': self.state_dict(),
        }