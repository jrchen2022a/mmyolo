import torch
from torch import nn
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaptiveLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=[2,3], keepdims=True)
        std = x.std(dim=[2,3], keepdims=True)
        x_normalized = (x - mean) / (std + self.eps)
        x_scaled = self.gamma.view(1, self.num_features, 1 ,1) *x_normalized + self.beta.view(1, self.num_features, 1, 1)
        return x_scaled

class SCAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCAM, self).__init__()
        self.pad1 = nn.ZeroPad2d(padding=(1, 1, 0, 0))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 1, 1))

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(1,3), padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(3,1), padding=0, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, padding=0, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.LN = AdaptiveLayerNorm(channel)


    def forward(self, x):
        b, c, h, w = x.size()
        pa1 = self.pad1(x)
        pa2 = self.pad2(x)
        cx1 = self.conv1(pa1)
        cx2 = self.conv2(pa2)

        max_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        max_pool_y = nn.AdaptiveMaxPool2d((1, w))
        avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        x_h = torch.cat([avg_pool_x(cx2),max_pool_x(cx2)],dim=1).view(b,c,h)
        x_w = torch.cat([avg_pool_y(cx1),max_pool_y(cx1)],dim=1).view(b,c,w)
        sa = self.sigmoid(torch.matmul(x_h.transpose(-1,-2), x_w))
        channel_weights = self.sigmoid(torch.matmul(x.view(b, c, h * w), sa.view(b, h * w, 1)).view(b, c, 1, 1))
        out = torch.cat((self.conv3(x) * sa.view(b, 1, h, w), x * channel_weights.expand_as(x)), 1)
        out = self.LN(self.conv4(out))

        return out + x
