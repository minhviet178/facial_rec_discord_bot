import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime

MODEL = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\yolo\yolov3-face.cfg'
WEIGHT = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\yolo\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

IMG_WIDTH, IMG_HEIGHT = 416, 416

def output(frame, save_path):

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    # Get frame dimension 
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
    # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # print(confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                topleft_x = int(center_x - width * 1/2) 
                topleft_y = int(center_y - width * 1/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    final_boxes = []
    result = frame.copy()
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        crop = result[top:top+height, left:left+width]

        # Bypass noise 
        fname = datetime.now().strftime('%m%d%H%M%S')
        save_image_path = os.path.join(save_path, f'{fname}_{i}.jpg')
        cv2.imwrite(save_image_path, crop)
        print(f'Successfully write at {save_image_path}')

import pathlib

#Extract all files from path    
img_name = 'babe'
path = f'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\preprocess_data\/{img_name}'
save_path = f'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\/test_data_30\/{img_name}'

# Get all paths from the folder -- Get only .jpg files -- Change as you want 
folder_lib = pathlib.Path(path)
file_names = sorted([str(item) for item in folder_lib.glob('*.jpg') if item.is_file()])

# Testing
for frame in file_names:
    image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
    output(image, save_path=save_path)