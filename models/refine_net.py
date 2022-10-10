import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class RefineNet(nn.Module):
    def __init__(self, num_residual_blocks):
        super(RefineNet, self).__init__()

        model = [nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 3, 3, padding=1),
            nn.LeakyReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, pyr_original_img):

        refined_output = self.model(pyr_original_img)

        return refined_output

def test_1layer():
    trans_High_model = RefineNet(num_residual_blocks=1)

    intput_t = torch.randn(1, 3, 224, 224)

    output_t = trans_High_model(intput_t)
    print(type(output_t))


    print(output_t.shape)

if __name__ == "__main__":
    #test()
    #test_1layer()
    test_1layer()