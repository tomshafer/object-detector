
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeError(Exception):
    pass


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
        x = self.conv(x)
        if self.batch_normalize:
            x = self.batchnorm(x)
        if self.activation:
            x = F.leaky_relu(x, negative_slope=0.1)
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
        if self.pad_a or self.pad_b:
            x = F.pad(x, (self.pad_a, self.pad_b, self.pad_a, self.pad_b), mode='constant', value=self.value)
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x


class YOLOv3Layer(nn.Module):
    # TODO: loss function
    def __init__(self, input_width, input_height, classes=80, anchors=[]):
        super().__init__()
        self.anchors = anchors
        self.classes = classes
        self.input_width = input_width
        self.input_height = input_height
    
    def forward(self, x):
        num_boxes = len(self.anchors)
        skip_length = (4+1+self.classes)
        if x.shape[1] != num_boxes*skip_length:
            raise ShapeError('no. filters != no. boxes * (5 + no. classes)')
        
        x = x.transpose(1, 2).transpose(2, 3)
        
        # Convert predictions to logistic where appropriate
        for box in range(num_boxes):
            xa, xb = box*skip_length, box*skip_length + 2
            x[:,:,:,xa:xb] = torch.sigmoid(x[:,:,:,xa:xb])
            xa, xb = box*skip_length+4, box*skip_length+4 + 1+self.classes
            x[:,:,:,xa:xb] = torch.sigmoid(x[:,:,:,xa:xb])
        
        # Do the box resizing
        for box in range(num_boxes):
            pw, ph = self.anchors[box]
            for row in range(x.shape[1]):
                for col in range(x.shape[2]):
                    offset = box*skip_length
                    x[:,row,col,offset+0] = (x[:,row,col,offset+0].item() + col)/x.shape[2]
                    x[:,row,col,offset+1] = (x[:,row,col,offset+1].item() + row)/x.shape[1]
                    x[:,row,col,offset+2] = pw/self.input_width*torch.exp(x[:,row,col,offset+2]).item()
                    x[:,row,col,offset+3] = ph/self.input_height*torch.exp(x[:,row,col,offset+3]).item()
        
        # Returns batch x num predictions x prediction
        # but only for nonzero scores
        x = x.contiguous().view(x.shape[0], -1, x.shape[-1]//num_boxes)
        mask = x[:,:,4] > 1e-6
        # https://discuss.pytorch.org/t/select-rows-of-the-tensor-whose-first-element-is-equal-to-some-value/1718/2
        return x[:,mask.squeeze(),:]


class TinyYOLOv3(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.w, self.h = width, height
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
            YOLOv3Layer(input_width=self.w, input_height=self.h,
                        classes=80, anchors=[(81, 82),  (135, 169),  (344, 319)])
        ])
        
        self.block4 = ConvolutionalLayer(256, 128, 1, 1, 0)
        
        self.block5 = nn.ModuleList([
            ConvolutionalLayer(384, 256, 3, 1, 1),
            ConvolutionalLayer(256, 255, 1, 1, 0, batch_normalize=False, activation=False),
            YOLOv3Layer(input_width=self.w, input_height=self.h,
                        classes=80, anchors=[(10, 14),  (23, 27),  (37, 58)])
        ])
    
    def forward(self, x):
        if self.training:
            self.images_seen += x.shape[0]
        
        # First block, with a checkpoint for the layer 8 exit
        for layer in self.block1:
            x = layer(x)
        x8 = x.clone()
        
        # Second block, with a checkpoint for the layer 13 exit
        for layer in self.block2:
            x = layer(x)
        x13 = x.clone()
        
        # First YOLO block
        for layer in self.block3:
            x = layer(x)
        y1 = x.clone()
        
        ## Route layer + conv + upscaling
        x = F.interpolate(self.block4(x13), scale_factor=2)
        
        # Concatenate w/layer 8 plus final YOLO layer
        x = torch.cat((x, x8), dim=1)
        for layer in self.block5:
            x = layer(x)
        
        return torch.cat((x, y1), dim=1)


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


## Helpers cf. Darknet
class Box(object):
    def __init__(self, xc, yc, w, h):
        assert w > 0
        assert h > 0
        self.x1 = xc-w/2
        self.x2 = xc+w/2
        self.y1 = yc-h/2
        self.y2 = yc+h/2
    
    def __repr__(self):
        return f'<Box(({self.x1}, {self.y1}), ({self.x2}, {self.y2}))>'
    
    @property
    def area(self):
        return (self.x2-self.x1)*(self.y2-self.y1)
    
    def intersection(self, other):
        if self.x1 > other.x2 or self.x2 < other.x1 \
        or self.y1 > other.y2 or self.y2 < other.y1:
            return 0
        
        w = min(self.x2, other.x2)-max(self.x1, other.x1)
        h = min(self.y2, other.y2)-max(self.y1, other.y1)
        return w*h
    
    def iou(self, other):
        i = self.intersection(other)
        u = self.area+other.area
        return i/u
    
    @staticmethod
    def from_tensor(t):
        s = t.data
        return Box.from_array(s)
    
    @staticmethod
    def from_array(a):
        return Box(*a[0:4])


class Detection(object):
    def __init__(self, t):
        # Get rid of torch stuff
        if len(t.shape) > 1:
            raise ShapeError('expected a 1-D tensor')
        
        s = np.array(t.data)
        self.classes = s.shape[0] - 5
        self.box = Box.from_array(s)
        self.objectness = s[4]
        self.class_prob = s[5:]
        self.prob = self.objectness*self.class_prob
    
    def __repr__(self):
        return f'<Detection({self.classes} classes, obj={self.objectness:.5f}, prob=[{self.prob[0]:.5f}, {self.prob[1]:.5f}, ...])>'


def get_yolo_detections(yolo_tensor, classes, thresh, orig_w, orig_h, scaled_w, scaled_h):
    num_dets = yolo_tensor.shape[0]
    if len(yolo_tensor.shape) > 2:
        raise ShapeError('get_detections only supports 1 image')
    if yolo_tensor.shape[-1]//(5+classes) != 1:
        raise ShapeError('bad tensor size maths or wrong no. classes')
    
    dets = []
    for i in range(num_dets):
        det = yolo_tensor[i]
        if det[4] < thresh:
            continue
        
        # Convert relative to absolute...x, w, y, h
        factor = scaled_w/orig_w if orig_w >= orig_h else scaled_h/orig_h
        
        delta = scaled_w - factor*orig_w
        det[0] = (scaled_w*det[0] - delta/2) * orig_w/(scaled_w-delta)
        det[2] = det[2] * (1 + delta/scaled_w) * orig_w
        
        delta = scaled_h - factor*orig_h
        det[1] = (scaled_h*det[1] - delta/2) * orig_h/(scaled_h-delta)
        det[3] = det[3] * (1 + delta/scaled_h) * orig_h
        
        det = Detection(det)
        det.prob = det.prob * (det.prob >= thresh)
        
        dets += [det]
    return dets
    

if __name__ == '__main__':
    import numpy as np
    import cv2

    net = TinyYOLOv3(width=416, height=416)
    load_weights('yolov3-tiny.weights', net)
    net.eval()
    
    #z = cv2.imread('random.png', cv2.IMREAD_COLOR)
    #z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
    #z = z.astype(np.float32)
    #z /= 255
    #z_final = torch.from_numpy(z.transpose(2, 0, 1))
    
    z = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
    output = z.copy()
    orig_h, orig_w = output.shape[:2]
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
    
    #output = np.array(z_final).transpose(1, 2, 0)
    #output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    #output *= 255
    #output = output.astype(np.uint8)
    
    # Predictions
    # # NMS, cf. box.c
    # predictions = sorted(range(y.shape[1]), key=lambda i: y[0,i,4], reverse=True)
    # for iclass in range(80):
    #     predictions = sorted(range(y.shape[1]), key=lambda i: y[0,i,5+iclass], reverse=True)
    #     for k in range(len(predictions)):
    #         if y[0,k,5+iclass] == 0:
    #             continue
    #         for l in range(k+1, len(predictions)):
    # 
    # 
    #print(predictions)
    #print(y[0,1262,4])
    
    
    detections = get_yolo_detections(y[0], 80, 0.5, orig_w, orig_h, 416, 416)
    for k in range(80):
        detections = sorted(detections, key=lambda d: d.class_prob[k], reverse=True)
        for i in range(len(detections)):
            di = detections[i]
            if di.class_prob[k] == 0:
                continue
            bi = di.box
            for j in range(i+1, len(detections)):
                dj = detections[j]
                bj = dj.box
                if bi.iou(bj) > 0.42:
                    dj.prob[k] = 0

    for det in detections:
        if max(det.prob) >= 0.5:
            print(det.objectness, end = '')
            for i,p in enumerate(det.prob):
                if p > 0:
                    print(f', ({i}, {p})', end='')
            print()
            cv2.rectangle(output, (int(det.box.x1), int(det.box.y1)), (int(det.box.x2), int(det.box.y2)), (255, 255, 0))
        
    cv2.imwrite('predictions.png', output)
    
