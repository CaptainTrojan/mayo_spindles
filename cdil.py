import torch
import torch.nn as nn
from numpy import ndarray

import sys

import matplotlib.pyplot as plt
import torchvision.ops
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, pad_mode):
        super(DeformableConv2d, self).__init__()

        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.ks = (kernel_size, 1)

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, self.ks, padding=self.padding,
                                     dilation=self.dilation, padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, kernel_size, self.ks, padding=self.padding, dilation=self.dilation,
                                        padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels, out_channels, self.ks, padding=self.padding, dilation=self.dilation,
                                      padding_mode=pad_mode, bias=False)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias, padding=self.padding, dilation=self.dilation,
                                          mask=modulator)
        return x


# One Conv. block
class Block(nn.Module):
    def __init__(self, model, c_in, c_out, ks, pad, dil, deformable):
        super(Block, self).__init__()
        self.model = model
        self.deform = deformable

        pad_mode = 'circular'

        if self.deform:
            self.conv = DeformableConv2d(c_in, c_out, ks, pad, dil, pad_mode)
        else:
            self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode=pad_mode))
            self.conv.weight.data.normal_(0, 0.01)
            self.conv.bias.data.normal_(0, 0.01)

        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.GELU()

    def forward(self, x):
        net = self.conv

        if self.deform:
            x_2d = x.unsqueeze(-1)
            out = net(x_2d)
            res = x if self.res is None else self.res(x)
            y = self.nonlinear(out) + res.unsqueeze(-1)
            return y.squeeze(-1)
        else:
            out = net(x)
            res = x if self.res is None else self.res(x)
            return self.nonlinear(out) + res


# Conv. blocks
class ConvPart(nn.Module):
    def __init__(self, model, hidden_channels, ks):
        super(ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)
        for i in range(1, num_layer):
            this_in = hidden_channels[i - 1]
            this_out = hidden_channels[i]
            this_dilation = 2 ** i
            this_padding = int(this_dilation * (ks - 1) / 2)
            if i < (num_layer - 3):
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, False)]
            else:
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, True)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


# Conv. + classifier
class CDIL(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels=32,
                 output_channels=256,
                 num_layers=4,
                 kernel_size=3,
                 ):
        super(CDIL, self).__init__()

        channels = [hidden_channels] * num_layers
        channels[0] = input_channels
        channels[-1] = output_channels

        self.conv = ConvPart('dict-cdil', channels, kernel_size, True, True)

    def forward(self, x):
        return self.conv(x)
