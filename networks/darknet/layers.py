
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import yolo


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
        self.pad_b = padding - 2*self.pad_a
    
    def forward(self, x):
        if self.pad_a or self.pad_b:
            x = F.pad(x, (self.pad_a, self.pad_b, self.pad_a, self.pad_b),
                      mode='constant', value=self.value)
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x


class YOLOv3Layer(nn.Module):
    def __init__(self, input_width, input_height, anchors, mask, classes=80, ignore_thresh=0.5, truth_thresh=1.0):
        super().__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.input_width = input_width
        self.input_height = input_height
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
    
    def forward(self, x, y=None):
        loss = -1
        num_boxes = len(self.mask)
        box_data = (4+1+self.classes)
        
        if x.shape[1] != num_boxes*box_data:
            raise utils.ShapeError('no. filters != no. boxes * (5 + no. classes)')
        
        # Swap channels to the end
        x = x.transpose(1, 2).transpose(2, 3)
        layer_h = x.shape[1]
        layer_w = x.shape[2]
        
        # Convert relevant predictions to logistic
        for box in range(num_boxes):
            xa, xb = box*box_data, box*box_data + 2
            x[:,:,:,xa:xb] = torch.sigmoid(x[:,:,:,xa:xb])
            xa, xb = box*box_data+4, box*box_data+4 + 1+self.classes
            x[:,:,:,xa:xb] = torch.sigmoid(x[:,:,:,xa:xb])
        
        # Rescale boxes to relative sizes
        for box in range(num_boxes):
            pw, ph = self.anchors[self.mask[box]]
            for row in range(layer_h):
                for col in range(layer_w):
                    offset = box*box_data
                    x[:,row,col,offset+0] = (x[:,row,col,offset+0].item() + col)/layer_w
                    x[:,row,col,offset+1] = (x[:,row,col,offset+1].item() + row)/layer_h
                    x[:,row,col,offset+2] = pw/self.input_width*torch.exp(x[:,row,col,offset+2]).item()
                    x[:,row,col,offset+3] = ph/self.input_height*torch.exp(x[:,row,col,offset+3]).item()
        
        # Loss
        if self.training:
            assert y is not None
            assert y.shape[0] == x.shape[0]
            losses = torch.zeros_like(x)
            
            # Step 1: fill with objectness scores relative to zero.
            # YOLO predicts x, y, w, h, o, classes.
            for image in range(x.shape[0]):
                truths = y[image]
                for row in range(layer_h):
                    for col in range(layer_w):
                        for i in range(num_boxes):
                            # Find the best truth via iou
                            with torch.no_grad():
                                offset = i*box_data
                                box = yolo.Box(*x[image,row,col,offset:offset+4])
                                best_t, best_iou = 0, 0
                                for t in range(truths.shape[0]):
                                    truth = yolo.Box(*truths[t, 0:4])
                                    iou = box.iou(truth)
                                    if iou > best_iou:
                                        best_t, best_iou = t, iou
                            if best_iou <= self.ignore_thresh:
                                losses[image, row, col, offset+4] = -x[image, row, col, offset+4]
                            # TODO: implement truth_thresh
                            if best_iou > self.truth_thresh:
                                raise NotImplementedError('< 1 truth_thresh not implemented')
            
            # Step 2: for each truth, update losses so that the correct box is assigned
            for image in range(x.shape[0]):
                truths = y[image]
                for t in range(truths.shape[0]):
                    with torch.no_grad():
                        truth = yolo.Box(*truths[t,0:4])
                        truth_shift = yolo.Box(0, 0, *truths[t,2:4])
                        # int() intentionally floors these numbers
                        row = int(truth.y*layer_h)
                        col = int(truth.x*layer_w)
                        best_n, best_iou = 0, 0
                        for i, (pw, ph) in enumerate(self.anchors):
                            pw /= self.input_width
                            ph /= self.input_height
                            prior_box = yolo.Box(0, 0, pw, ph)
                            iou = truth_shift.iou(prior_box)
                            if iou > best_iou:
                                best_n = i
                                best_iou = iou
                    
                    if best_n not in self.mask:
                        continue
                    
                    mask_index = self.mask.index(best_n)
                    offset = mask_index*box_data
                    
                    # Box delta
                    pw, ph = self.anchors[best_n]
                    tx = truth.x*layer_w-col
                    ty = truth.y*layer_h-row
                    tw = torch.log(truth.w*self.input_width/pw)
                    th = torch.log(truth.h*self.input_height/ph)
                    
                    # TODO: why (2 - w*h)?
                    losses[image, row, col, offset+0] = (2-truth.w*truth.h)*(tx - x[image, row, col, offset+0])
                    losses[image, row, col, offset+1] = (2-truth.w*truth.h)*(ty - x[image, row, col, offset+1])
                    losses[image, row, col, offset+2] = (2-truth.w*truth.h)*(tw - x[image, row, col, offset+2])
                    losses[image, row, col, offset+3] = (2-truth.w*truth.h)*(th - x[image, row, col, offset+3])
                    
                    # Objectness delta
                    losses[image, row, col, offset+4] = 1 - x[image, row, col, offset+4]
                    
                    # Class delta
                    class_index = int(truths[t,4])
                    xa, xb = offset+4+1, offset+4+1 + self.classes
                    if losses[image, row, col, xa+class_index] != 0:
                        raise NotImplementedError('< 1 truth_thresh not implemented')
                    losses[image, row, col, xa:xb] = -x[image, row, col, xa:xb]
                    losses[image, row, col, xa+class_index] = 1-x[image, row, col, xa+class_index]
            
            loss = torch.sum(torch.pow(losses, 2))
        return x.contiguous().view(x.shape[0], -1, 4+1+self.classes), loss
