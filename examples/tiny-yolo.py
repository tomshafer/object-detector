
import os
import sys
sys.path.append('..')

import cv2
import numpy as np
import torch

from urllib import request
from networks.darknet import yolo


if __name__ == '__main__':
    # Download YOLO v3 weights
    base_dir = os.path.dirname(__file__)
    weights_file = os.path.join(base_dir, 'yolov3-tiny.weights')
    if not os.path.exists(weights_file):
        while True:
            response = input('Cannot find weights file. Download? (y/n) ')
            if response.strip().lower()[0] == 'y':
                print('Downloading...')
                request.urlretrieve(
                    'https://pjreddie.com/media/files/yolov3-tiny.weights',
                    weights_file)
                break
            elif response.strip().lower()[0] == 'n':
                quit()
    
    # Run YOLO on the example image
    input = yolo.load_image_cv2(os.path.join(base_dir, 'dog.jpg'))
    orig_h, orig_w = input.shape[:2]
    output_img = input[:]
    
    input = yolo.letterbox_input_cv2(input, 416, 416)
    input = torch.tensor(input.transpose(2, 0, 1))
    
    model = yolo.TinyYOLOv3(416, 416, weights='yolov3-tiny.weights')
    model.eval()
    with torch.no_grad():
        output, loss = model(input.unsqueeze(0))
        detections = yolo.get_yolo_detections(output[0], 0.5)
        detections = yolo.nms_yolo_detections(detections, 80, prob_thresh=0.5)
        detections = yolo.rescale_yolo_predictions(detections, orig_w, orig_h)
    
    # Convert to OpenCV format, add rectangles, etc.
    output_img *= 255
    output_img = output_img.astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    np.random.seed(1024)
    for det in detections:
        print('Class {:>2d}: {:.0f}%'.format(
            det.most_probable_class, 100*det.probs[det.most_probable_class]))
        # http://answers.opencv.org/question/185393/typeerror-scalar-value-for-argument-color-is-not-numeric/
        color = tuple(map(
            lambda x: np.asscalar(np.uint8(x)),
            np.random.randint(0, 255, 3)))
        output_img = cv2.rectangle(
            output_img, det.box.corner_int('tl'), det.box.corner_int('br'),
            color, 2)
        output_img = cv2.putText(
            output_img, '{:.0f}%'.format(100*det.max_probability),
            det.box.corner_int('tl'), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    cv2.imwrite('predictions.png', output_img)
