import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(          
            # # 512x512
            # nn.Conv2d(1, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # # 256x256
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # 150x150
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 75x75
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1),
            # 38x38
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 19x19
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1),
            # 10x10
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 8x8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 4x4
            nn.Conv2d(64, 1, kernel_size=3, padding=0),
            # 2x2
        )

    def forward(self, x):
        # x = self.layers(x).squeeze()
        # out = self.out_layer(x)

        # return out
        return self.layers(x).sum(axis=2).squeeze() #.reshape(-1, 25).squeeze() # return as [[[1,2,3,4,5,6,7,8,9]]]

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