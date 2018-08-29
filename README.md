# YOLO PyTorch Implementation

This is a PyTorch implementation of [YOLO][] version 3.

## Requirements

 - Python 3
 - PyTorch
 - OpenCV
 - Numpy

## FAQ

#### Does it train?

Yes, actually. Slowly—I haven't worked out how to vectorize the YOLO loss computation—but it trains. At least it overfits the YOLO Dog Image(TM).

#### Can it load Darknet weights?

Yep.

#### What models are implemented?

 - [Tiny YOLO v3][tiny-yolo-cfg]

I'm going to add more. Maybe a full Darknet-style thing, maybe not. But the whole point of this thing is to easily swap feature extractors in and out.


[YOLO]: https://pjreddie.com/darknet/yolo/
[tiny-yolo-cfg]: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
