import torch
from torch import nn
from loss_function import gram
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
    


#这里return gen_img()要学会理解，返回的其实是self.weight，即forward函数的返回值