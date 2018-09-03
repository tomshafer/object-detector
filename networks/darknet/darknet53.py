
from torch import nn
from torch.nn import functional as F

from . import layers as ls


class _ResidualBlock(nn.Module):
    """Darknet 53-style residual block.

    Two convolutional layers, each with Darknet-style batch normalization
    and leaky ReLU activation.
    """
    def __init__(self, in_channels, layer1_channels, layer2_channels):
        super().__init__()
        self.conv1 = ls.ConvolutionalLayer(in_channels, layer1_channels, 1, 1, 0)
        self.conv2 = ls.ConvolutionalLayer(layer1_channels, layer2_channels, 3, 1, 1)

    def forward(self, x):
        res = x.clone()
        x = self.conv2(self.conv1(x))
        return x+res


class Darknet53(nn.Module):
    """Darknet-53

    See: <https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg>.

    Seems deisgned for 256-px images.

    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers += [ls.ConvolutionalLayer(3,  32, 3, 1, 1)]
        self.layers += [ls.ConvolutionalLayer(32, 64, 3, 2, 1)]
        self.layers += [_ResidualBlock(64, 32, 64)]
        self.layers += [ls.ConvolutionalLayer(64, 128, 3, 2, 1)]
        self.layers += 2 * [_ResidualBlock(128, 64, 128)]
        self.layers += [ls.ConvolutionalLayer(128, 256, 3, 2, 1)]
        self.layers += 8 * [_ResidualBlock(256, 128, 256)]
        self.layers += [ls.ConvolutionalLayer(256, 512, 3, 2, 1)]
        self.layers += 8 * [_ResidualBlock(512, 256, 512)]
        self.layers += [ls.ConvolutionalLayer(512, 1024, 3, 2, 1)]
        self.layers += 4 * [_ResidualBlock(1024, 512, 1024)]
        self.predict_layer = ls.ConvolutionalLayer(1024, 1000, 1, 1, 0, False, False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        print(x.shape)
        x = F.avg_pool2d(x, x.shape[2:])
        x = self.predict_layer(x)
        return x.view(x.shape[0], -1)
