
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            batch_normalize=True,
            activation=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_normalize = batch_normalize
        self.activation = activation
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(not batch_normalize))
        
        if batch_normalize:
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                momentum=0.01,
                eps=1e-6)

    def forward(self, x):
        print(self)
        print('x', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, l], end=', ')
        print()
        print('x', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, 4001+l], end=', ')
        print()
        print()
        
        x = self.conv(x)
        
        print('a', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, l], end=', ')
        print()
        print('a', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, 4001+l], end=', ')
        print()
        print()
        
        if self.batch_normalize:
            x = self.batchnorm(x)
        
        print('b', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, l], end=', ')
        print()
        print('b', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, 4001+l], end=', ')
        print()
        print()
        
        if self.activation:
            x = F.leaky_relu(x, negative_slope=0.1)
        
        print('c', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, l], end=', ')
        print()
        print('c', end='  ')
        for l in range(10):
            print('%e' % x.reshape(x.shape[0], -1)[0, 4001+l], end=', ')
        print()
        print()
        
        return x


class MaxPoolLayer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            value=-1e38
    ):
        super().__init__()
        self.value = value
        self.kernel_size = kernel_size
        self.stride = stride
        
        # this is from maxpool_layer.c -- we pad the right/bottom preferentially
        self.pad_a = padding//2
        self.pad_b = padding - 2*(padding//2)
    
    def forward(self, x):
        #print(self)
        if self.pad_a or self.pad_b:
            #print(x.shape)
            x = F.pad(x, (self.pad_a, self.pad_b, self.pad_a, self.pad_b), mode='constant', value=self.value)
            #print(x.shape)
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        #print(x.shape)
        return x


class YOLOv3Layer(nn.Module):
    # TODO: loss function
    def __init__(self, classes=80, anchors=[]):
        super().__init__()
        self.anchors = anchors
        self.classes = classes
    
    def forward(self, x):
        #print(self)
        assert x.shape[1] == len(self.anchors)*(4+1+self.classes)
        
        # Undo reshape.
        # Return batch x 3 x 255 after box conversion
        # Then we can join with other predictions
        
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3, -1)
        # Convert predictions to logistic where appropriate
        x[:,:,:,:,0:2]  = torch.sigmoid(x[:,:,:,:,0:2])
        x[:,:,:,:,4:] = torch.sigmoid(x[:,:,:,:,4:])
        return x


class TinyYOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.w, self.h = None, None
        self.images_seen = 0
        
        self.block1 = nn.ModuleList([
            ConvolutionalLayer(3,    16,  3, 1, 1), MaxPoolLayer(2, 2),
            ConvolutionalLayer(16,   32,  3, 1, 1), MaxPoolLayer(2, 2),
            ConvolutionalLayer(32,   64,  3, 1, 1), MaxPoolLayer(2, 2),
            ConvolutionalLayer(64,   128, 3, 1, 1), MaxPoolLayer(2, 2),
            ConvolutionalLayer(128,  256, 3, 1, 1)
        ])
        
        self.block2 = nn.ModuleList([
            MaxPoolLayer(2, 2),
            ConvolutionalLayer(256,  512,  3, 1, 1), MaxPoolLayer(2, 1, 1),
            ConvolutionalLayer(512,  1024, 3, 1, 1),
            ConvolutionalLayer(1024, 256,  1, 1, 0)
        ])
        
        self.block3 = nn.ModuleList([
            ConvolutionalLayer(256, 512, 3, 1, 1),
            ConvolutionalLayer(512, 255, 1, 1, 0, batch_normalize=False, activation=False),
            YOLOv3Layer(classes=80, anchors=[(81, 82),  (135, 169),  (344, 319)])
        ])
        
        self.block4 = ConvolutionalLayer(256, 128, 1, 1, 0)
        
        self.block5 = nn.ModuleList([
            ConvolutionalLayer(384, 256, 3, 1, 1),
            ConvolutionalLayer(256, 255, 1, 1, 0, batch_normalize=False, activation=False),
            YOLOv3Layer(classes=80, anchors=[(10, 14),  (23, 27),  (37, 58)])
        ])
    
    def forward(self, x):
        # First block, with a checkpoint for the layer 8 exit
        for layer in self.block1:
            x = layer(x)
        #x8 = x.clone()
        
        # Second block, with a checkpoint for the layer 13 exit
        for layer in self.block2:
            x = layer(x)
        #x13 = x.clone()
        
        # First YOLO block
        for layer in self.block3:
            x = layer(x)
        
        ## Route layer + conv + upscaling
        #x = F.interpolate(self.block4(x13), scale_factor=2)
        #
        ## Concatenate w/layer 8 plus final YOLO layer
        #x = torch.cat((x, x8), dim=1)
        #for layer in self.block5:
        #    x = layer(x)
        
        return x


def _load_into_tensor(tensor, weights, offset, size):
    tensor.data = torch.from_numpy(weights[offset:offset+size]).view_as(tensor.data)
    return offset + size


def load_weights(file, network):
    with open(file, 'rb') as fil:
        major, minor, revision = np.frombuffer(fil.read(12), dtype=np.uint32)
        if (10*major + minor) >= 2 and major < 1000 and minor < 1000:
            network.images_seen = int(np.frombuffer(fil.read(8), dtype=np.uint64)[0])
        else:
            network.images_seen = int(np.frombuffer(fil.read(4), dtype=np.uint32)[0])
        weights = np.fromfile(fil, dtype=np.float32)

    offset = 0
    for module in net.modules():
        if module.__class__.__name__ != 'ConvolutionalLayer':
            continue
        c = module.in_channels
        n = module.out_channels
        k = module.kernel_size
        if module.batch_normalize:
            offset = _load_into_tensor(module.batchnorm.bias,         weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.weight,       weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.running_mean, weights, offset, n)
            offset = _load_into_tensor(module.batchnorm.running_var,  weights, offset, n)
        else:
            offset = _load_into_tensor(module.conv.bias, weights, offset, n)
        offset = _load_into_tensor(module.conv.weight, weights, offset, n * c * k**2)
    
    if offset != weights.size:
        print('Warning: offset != weights.size in load_weights.',
              'offset = {}, weights.size = {}'.format(offset, weights.size))


if __name__ == '__main__':
    import numpy as np
    import cv2

    net = TinyYOLOv3()    
    load_weights('yolov3-tiny.weights', net)
    net.eval()
    
    #z = cv2.imread('random.png', cv2.IMREAD_COLOR)
    #z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
    #z = z.astype(np.float32)
    #z /= 255
    #z_final = torch.from_numpy(z.transpose(2, 0, 1))
    
    z = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
    #z = cv2.imread('sized.png', cv2.IMREAD_COLOR)
    z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
    hscale = 416/z.shape[1]
    z = cv2.resize(z, (416, int(round(z.shape[0]*hscale))), interpolation=cv2.INTER_CUBIC)
    z = z.astype(np.float32)
    z /= 255
    # Embed in a square
    offset = (416-312)//2
    z_final = 0.5*np.ones((416, 416, 3), dtype=np.float32)
    z_final[offset:offset+312,:,:] = z
    #z_final = z
    z_final = torch.from_numpy(z_final.transpose(2, 0, 1))
    
    y = net(z_final.unsqueeze(0))
    
    # Predictions
    predictions = []
    thresh = 0.5
    y.shape
    for item in range(y.shape[0]):
        for row in range(y.shape[1]):
            for col in range(y.shape[2]):
                for n in range(y.shape[3]):
                    tx, ty, tw, th, to = y[item, row, col, n, 0:5]
                    if to < thresh:
                        continue
                    print(item, row, col, n, to)

    #dir(net.block3[-1])
