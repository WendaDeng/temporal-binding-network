import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalFusion(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(TemporalFusion, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 128, kernel_size=1)
        
        self.branch3x3_1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_2a = conv_block(320, 320, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(320, 320, kernel_size=(3, 1), padding=(1, 0))

        self.branch5x5_1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch5x5_2a = conv_block(320, 320, kernel_size=(1, 5), padding=(0, 2))
        self.branch5x5_2b = conv_block(320, 320, kernel_size=(5, 1), padding=(2, 0))

        self.branch7x7_1 = conv_block(in_channels, 256, kernel_size=1)
        self.branch7x7_2a = conv_block(256, 256, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_2b = conv_block(256, 256, kernel_size=(7, 1), padding=(3, 0))

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2a(branch3x3)
        branch3x3 = self.branch3x3_2b(branch3x3) 

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2a(branch5x5)
        branch5x5 = self.branch5x5_2b(branch5x5)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2a(branch7x7)
        branch7x7 = self.branch7x7_2b(branch7x7)

        outputs = [branch1x1, branch3x3, branch5x5, branch7x7]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)