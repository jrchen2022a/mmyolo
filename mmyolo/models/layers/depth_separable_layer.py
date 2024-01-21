from torch import nn


class DepthSeparableLayer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, padding=0, bias=False, BatchNorm=nn.BatchNorm2d):
        super(DepthSeparableLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
