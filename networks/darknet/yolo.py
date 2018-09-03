"""YOLO models and related helpers.

The actual YOLOv3 layer is in darknet/layers.py.
"""

import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet53 import _ResidualBlock
from . import layers as ls
from . import utils


class Box(object):
    def __init__(self, xc, yc, w, h):
        assert w > 0
        assert h > 0
        # Internal representation is corners
        self.x = xc
        self.y = yc
        self.w = w
        self.h = h

    def get_corner_coords(self):
        return (self.x-self.w/2, self.y-self.h/2), (self.x+self.w/2, self.y+self.h/2)

    def __repr__(self):
        (x1, y1), (x2, y2) = self.get_corner_coords()
        return f'<Box(({x1:.3f}, {y1:.3f}), ({x2:.3f}, {y2:.3f}))>'

    @property
    def area(self):
        return self.w*self.h

    def intersection(self, other):
        (sx1, sy1), (sx2, sy2) = self.get_corner_coords()
        (ox1, oy1), (ox2, oy2) = other.get_corner_coords()
        if sx1 > ox2 or sx2 < ox1 or sy1 > oy2 or sy2 < oy1:
            return 0
        w = min(sx2, ox2)-max(sx1, ox1)
        h = min(sy2, oy2)-max(sy1, oy1)
        return w*h

    def iou(self, other):
        i = self.intersection(other)
        u = self.area+other.area-i
        return i/u

    def corner_int(self, pos='tl'):
        if pos == 'tl':
            (x1, y1), _ = self.get_corner_coords()
            return (int(x1), int(y1))
        if pos == 'br':
            _, (x2, y2) = self.get_corner_coords()
            return (int(x2), int(y2))
        raise ValueError(f'unknown position: "{pos}"')


class Detection(object):
    def __init__(self):
        self.box = None
        self.objectness = None
        self.classes = []
        self.probs = []

    def __repr__(self):
        return (
            f'<Detection(obj={self.objectness:.5f} '
            f'class={self.most_probable_class:d} '
            f'prob={self.probs[self.most_probable_class]:.5f})>')

    @property
    def most_probable_class(self):
        return self.classes.index(max(self.classes))

    @property
    def max_probability(self):
        return max(self.probs)

    @staticmethod
    def from_tensor(tensor):
        assert len(tensor.shape) == 1
        bx, by, bw, bh, bo = tensor[:5].tolist()
        classes = tensor[5:].tolist()
        det = Detection()
        det.box = Box(bx, by, bw, bh)
        det.objectness = bo
        det.classes = classes
        det.probs = [bo*c for c in classes]
        return det


class TinyYOLOv3(nn.Module):
    """
    This is Tiny YOLO v3 as defined at
    <https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg>.
    The first set of anchors goes to the 13x13 detections layer; the second set
    goes to the 26x26 layer. YOLO v3's 'masking' is calculated from the list.
    """
    def __init__(
            self,
            input_width,
            input_height,
            anchors=[
                [(81, 82), (135, 169), (344, 319)],
                [(10, 14), (23, 27), (37, 58)]
            ],
            ignore_thresh=0.7,
            truth_thresh=1.0,
            classes=80,
            weights=None
    ):
        super().__init__()
        self.input_w, self.input_h = input_width, input_height
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.classes = classes
        self.images_seen = 0

        # Set masking indices for the flat anchor list.
        self.anchors, self.masks = _flatten_anchors(anchors)

        self.block1 = nn.ModuleList([
            ls.ConvolutionalLayer(3,    16,  3, 1, 1), ls.MaxPoolLayer(2, 2),
            ls.ConvolutionalLayer(16,   32,  3, 1, 1), ls.MaxPoolLayer(2, 2),
            ls.ConvolutionalLayer(32,   64,  3, 1, 1), ls.MaxPoolLayer(2, 2),
            ls.ConvolutionalLayer(64,   128, 3, 1, 1), ls.MaxPoolLayer(2, 2),
            ls.ConvolutionalLayer(128,  256, 3, 1, 1)
        ])

        self.block2 = nn.ModuleList([
            ls.MaxPoolLayer(2, 2),
            ls.ConvolutionalLayer(256,  512,  3, 1, 1), ls.MaxPoolLayer(2, 1, 1),
            ls.ConvolutionalLayer(512,  1024, 3, 1, 1),
            ls.ConvolutionalLayer(1024, 256,  1, 1, 0)
        ])

        self.block3 = nn.ModuleList([
            ls.ConvolutionalLayer(256, 512, 3, 1, 1),
            ls.ConvolutionalLayer(512, 255, 1, 1, 0, batch_normalize=False, activation=False),
            ls.YOLOv3Layer(input_width=self.input_w, input_height=self.input_h,
                          classes=self.classes, anchors=self.anchors, mask=self.masks[0],
                          ignore_thresh=self.ignore_thresh, truth_thresh=self.truth_thresh)
        ])

        self.block4 = ls.ConvolutionalLayer(256, 128, 1, 1, 0)

        self.block5 = nn.ModuleList([
            ls.ConvolutionalLayer(384, 256, 3, 1, 1),
            ls.ConvolutionalLayer(256, 255, 1, 1, 0, batch_normalize=False, activation=False),
            ls.YOLOv3Layer(input_width=self.input_w, input_height=self.input_h,
                          classes=self.classes, anchors=self.anchors, mask=self.masks[1],
                          ignore_thresh=self.ignore_thresh, truth_thresh=self.truth_thresh)
        ])

        if weights is not None:
            utils.load_weights(weights, self)

    def forward(self, x, y=None):
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

        # First YOLO layer
        for layer in self.block3[:-1]:
            x = layer(x)
        yolo_out1, yolo_loss1 = self.block3[-1](x, y)

        # Route layer + conv + upscaling
        x = F.interpolate(self.block4(x13), scale_factor=2)
        # Concatenate w/layer 8
        x = torch.cat((x, x8), dim=1)

        # Final YOLO layer
        for layer in self.block5[:-1]:
            x = layer(x)
        yolo_out2, yolo_loss2 = self.block5[-1](x, y)

        yolo_out = torch.cat((yolo_out1, yolo_out2), dim=1)
        yolo_loss = yolo_loss1+yolo_loss2
        return (yolo_out, yolo_loss)


class YOLOv3(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            anchors=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)]
            ],
            ignore_thresh=0.7,
            truth_thresh=1.0,
            classes=80,
            weights=None
    ):
        super().__init__()
        self.input_w, self.input_h = input_width, input_height
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.classes = classes
        self.images_seen = 0
        # Set masking indices for the flat anchor list.
        self.anchors, self.masks = _flatten_anchors(anchors)

        self.block1 = nn.ModuleList([
            ls.ConvolutionalLayer(3,  32, 3, 1, 1),     # Layer 0
            ls.ConvolutionalLayer(32, 64, 3, 2, 1),
            _ResidualBlock(64, 32, 64),                 # Layer 4
            ls.ConvolutionalLayer(64, 128, 3, 2, 1),
            _ResidualBlock(128, 64, 128),
            _ResidualBlock(128, 64, 128),               # Layer 11
            ls.ConvolutionalLayer(128, 256, 3, 2, 1),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256),
            _ResidualBlock(256, 128, 256)               # Layer 36
        ])

        self.block2 = nn.ModuleList([
            ls.ConvolutionalLayer(256, 512, 3, 2, 1),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512),
            _ResidualBlock(512, 256, 512)               # Layer 61
        ])

        self.block3 = nn.ModuleList([
            ls.ConvolutionalLayer(512, 1024, 3, 2, 1),
            _ResidualBlock(1024, 512, 1024),
            _ResidualBlock(1024, 512, 1024),
            _ResidualBlock(1024, 512, 1024),
            _ResidualBlock(1024, 512, 1024),            # Layer 74
            ls.ConvolutionalLayer(1024, 512, 1, 1, 0),
            ls.ConvolutionalLayer(512, 1024, 3, 1, 1),
            ls.ConvolutionalLayer(1024, 512, 1, 1, 0),
            ls.ConvolutionalLayer(512, 1024, 3, 1, 1),
            ls.ConvolutionalLayer(1024, 512, 1, 1, 0)   # Layer 79
        ])

        self.yolo1 = nn.ModuleList([
            ls.ConvolutionalLayer(512, 1024, 3, 1, 1),
            ls.ConvolutionalLayer(1024, 255, 1, 1, 0, False, False),
            ls.YOLOv3Layer(
                input_width=self.input_w,
                input_height=self.input_h,
                classes=self.classes,
                anchors=self.anchors,
                mask=self.masks[0],
                ignore_thresh=self.ignore_thresh,
                truth_thresh=self.truth_thresh)        # Layer 82
        ])

        self.block4a = ls.ConvolutionalLayer(512, 256, 1, 1, 0)

        self.block4b = nn.ModuleList([
            ls.ConvolutionalLayer(768, 256,  1, 1, 0),
            ls.ConvolutionalLayer(256, 512, 3, 1, 1),
            ls.ConvolutionalLayer(512, 256, 1, 1, 0),
            ls.ConvolutionalLayer(256, 512, 3, 1, 1),
            ls.ConvolutionalLayer(512, 256, 1, 1, 0)    # Layer 91
        ])

        self.yolo2 = nn.ModuleList([
            ls.ConvolutionalLayer(256, 512, 3, 1, 1),
            ls.ConvolutionalLayer(512, 255, 1, 1, 0, False, False),
            ls.YOLOv3Layer(
                input_width=self.input_w,
                input_height=self.input_h,
                classes=self.classes,
                anchors=self.anchors,
                mask=self.masks[1],
                ignore_thresh=self.ignore_thresh,
                truth_thresh=self.truth_thresh)         # Layer 94
        ])

        self.block5a = ls.ConvolutionalLayer(256, 128, 1, 1, 0)

        self.block5b = nn.ModuleList([
            ls.ConvolutionalLayer(384, 128,  1, 1, 0),
            ls.ConvolutionalLayer(128, 256, 3, 1, 1),
            ls.ConvolutionalLayer(256, 128, 1, 1, 0),
            ls.ConvolutionalLayer(128, 256, 3, 1, 1),
            ls.ConvolutionalLayer(256, 128, 1, 1, 0)
        ])

        self.yolo3 = nn.ModuleList([
            ls.ConvolutionalLayer(128, 256, 3, 1, 1),
            ls.ConvolutionalLayer(256, 255, 1, 1, 0, False, False),
            ls.YOLOv3Layer(
                input_width=self.input_w,
                input_height=self.input_h,
                classes=self.classes,
                anchors=self.anchors,
                mask=self.masks[2],
                ignore_thresh=self.ignore_thresh,
                truth_thresh=self.truth_thresh)         # Layer 106
        ])

        if weights is not None:
            utils.load_weights(weights, self)

    def forward(self, x, y=None):
        if self.training:
            self.images_seen += x.shape[0]

        for layer in self.block1:
            x = layer(x)
        layer_36 = x.clone()

        for layer in self.block2:
            x = layer(x)
        layer_61 = x.clone()

        for layer in self.block3:
            x = layer(x)
        layer_79 = x.clone()

        # YOLO 1
        for layer in self.yolo1[:-1]:
            x = layer(x)
        yolo_out1, yolo_loss1 = self.yolo1[-1](x, y)

        # Route, upscale, conv, YOLO 2
        x = F.interpolate(self.block4a(layer_79), scale_factor=2)
        x = torch.cat((x, layer_61), dim=1)

        for layer in self.block4b:
            x = layer(x)
        layer_91 = x.clone()

        for layer in self.yolo2[:-1]:
            x = layer(x)

        yolo_out2, yolo_loss2 = self.yolo2[-1](x, y)

        # Route, upscale, conv, YOLO 3
        x = F.interpolate(self.block5a(layer_91), scale_factor=2)
        x = torch.cat((x, layer_36), dim=1)

        for layer in self.block5b:
            x = layer(x)

        for layer in self.yolo3[:-1]:
            x = layer(x)

        yolo_out3, yolo_loss3 = self.yolo3[-1](x, y)

        yolo_out = torch.cat((yolo_out1, yolo_out2, yolo_out3), dim=1)
        yolo_loss = yolo_loss1+yolo_loss2+yolo_loss3
        return (yolo_out, yolo_loss)


def get_yolo_detections(tensor, thresh=0.5):
    if len(tensor.shape) > 2:
        raise utils.ShapeError('can only get detections for a single image')
    detections = []
    for t in tensor:
        det = Detection.from_tensor(t)
        if det.max_probability > thresh:
            detections += [det]
    return detections


def nms_yolo_detections(detections, classes, iou_thresh=0.45, prob_thresh=None):
    detections = copy.deepcopy(detections)
    for c in range(classes):
        detections = sorted(detections, key=lambda d: d.probs[c], reverse=True)
        for ia, det_a in enumerate(detections):
            if det_a.probs[c] == 0:
                continue
            for ib in range(ia+1, len(detections)):
                det_b = detections[ib]
                if det_a.box.iou(det_b.box) > iou_thresh:
                    det_b.probs[c] = 0
    if prob_thresh is not None:
        detections = [det for det in detections if det.max_probability > prob_thresh]
    return detections


def rescale_yolo_predictions(detections, orig_w, orig_h):
    """Undo scaling and letterboxing.

    **Assumes ONE image and a SQUARE box.**
    """
    detections = copy.deepcopy(detections)
    aspect = orig_w/orig_h
    for det in detections:
        box = det.box
        if aspect > 1:
            # Wider than tall -- vertical boxes need to be adjusted
            box.x *= orig_w
            box.y = box.y*orig_w - (orig_w-orig_h)/2
            box.w *= orig_w
            box.h *= orig_h * (2 - orig_h/orig_w)
        else:
            # Taller than wide -- horizontal boxes need to be adjusted
            box.y *= orig_h
            box.h *= orig_h
            box.w *= orig_w * (2 - orig_w/orig_h)
            box.x = box.x*orig_h - (orig_h-orig_w)/2
    return detections


def load_image_cv2(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255
    return image


def load_truths(file):
    with open(file, 'r', encoding='utf-8') as fil:
        truths = []
        for line in fil:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            line = line.split()
            cl = int(line[0])
            xc, yc, w, h = map(float, line[1:])
            truths += [xc, yc, w, h, cl]
    truths = np.array(truths, dtype=np.float32)
    return truths.reshape(-1, 5)


def letterbox_input_cv2(image, out_w, out_h, truths=None):
    if image.shape[-1] != 3:
        raise utils.ShapeError('expected CV2 channels-last format')

    image_h, image_w = image.shape[0:2]
    scale = min(out_h/image_h, out_w/image_w)

    # Resize and letterbox the image
    tmp_w = int(scale*image_w)
    tmp_h = int(scale*image_h)
    image = cv2.resize(image, (tmp_w, tmp_h), interpolation=cv2.INTER_CUBIC)
    image[image < 0] = 0
    image[image > 1] = 1

    # Embed in the letterbox
    letterboxed = 0.5*np.ones((out_h, out_w, 3), dtype=np.float32)
    delta_w = out_w-tmp_w
    delta_h = out_h-tmp_h
    letterboxed[
        delta_h//2:(delta_h//2 + tmp_h),
        delta_w//2:(delta_w//2 + tmp_w),
        :] = image

    # Rescale the ground truths, depending on which dimension is squished.
    # The heights/widths will be squished, and the centroids will be moved.
    if truths is not None:
        assert delta_w == 0 or delta_h == 0
        if delta_h > 0:
            truths[:, 3] = truths[:, 3]*tmp_h/out_h
            truths[:, 1] = (truths[:, 1]*tmp_h + delta_h/2) / out_h
        else:
            truths[:, 2] = truths[:, 2]*tmp_w/out_w
            truths[:, 0] = (truths[:, 0]*tmp_w + delta_w/2) / out_w
        return letterboxed, truths
    return letterboxed


def _flatten_anchors(anchor_lists):
    """Return a flattened set of anchor boxes.

    Take an iterable of iterables (a list of anchor coordinates) and return
        (1) a flattened list of anchors and
        (2) lists of "mask" indices mapping the flattened anchor list onto the
            original list.

    Arguments:
        anchor_lists {iterable} -- list of lists of anchor boxes per YOLO layer

    """
    # Generate the mask lists
    masks = []
    mask_count = 0
    for group in anchor_lists:
        this_mask_set = []
        for _ in group:
            this_mask_set.append(mask_count)
            mask_count += 1
        masks.append(this_mask_set)

    # Flatten the anchor list
    anchors = [a for mask in anchor_lists for a in mask]

    return (anchors, masks)
