import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import pathlib
import cv2

# Define YOLOv3
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

# Intructions
st.markdown("<h1 style='text-align: center; color: black;'>FACE REGISTRATION</h1>", unsafe_allow_html=True)
st.header("1. Capture a video of yourself")
st.subheader("- Take a short 3 - 4 seconds video of yourself")
st.subheader("- Look straight into the camera")

st.header('2. Convert your video into .mp4 video')
st.subheader('- Our system only accepts .mp4 files')
st.subheader('- Click [here](https://cloudconvert.com/mp4-converter) to use an online converter of our choice!')

# Enter info
st.header("3. Enter your personal information")
path = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master'
preprocess_path = os.path.join(path, 'preprocess_data')
col1, col2 = st.beta_columns(2)
with col1:
    name = st.text_input('Full Name')


with col2:
    course = st.multiselect('Course Name',['Free Python Course', 'Data Analytics', 'Full-Stack Web Development', 'Machine Learning Engineer', 'Corporate Training'] )

# Upload video
st.header('4. Upload your .mp4 video')
save_folder = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\Submitted_Video'
save_path = os.path.join(save_folder, f'{name}.mp4')
video = st.file_uploader('Choose an mp4 file', ['mp4'])

# Check if the video is submitted
if video != None:
    try:
        with open(save_path,"wb") as f:
            f.write(video.getbuffer())
        st.write('File uploaded succesfully!')
    except:
        st.write('File failed to upload. Please try again')



if st.button('Submit'):
    # Check if the info are given
    if name == None:
        st.write('Full Name Missing!')
    elif course == None:
        st.write('Course Name Missing!')
    else:
        st.write('Video submitted!')
    # # Extract frames from submitted video
    #     vidcap = cv2.VideoCapture(os.path.join(path, f'Submitted_Video\{name}.mp4'))
    #     def getFrame(sec):
    #         # Reads the submitted video
    #         vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    #         hasFrames,image = vidcap.read()
    #         # Extract and save all frames to the preprocess data folder 
    #         if hasFrames:
    #             dir = name
    #             new_dir = os.path.join(preprocess_path, dir)
    #             print(new_dir)
    #             if os.path.isfile(new_dir) == True:
    #                 st.write('Username already exists, please try another name')
    #             else:
    #                 os.mkdir(new_dir)
    #             cv2.imwrite(os.path.join(new_dir, f"{name}"+str(count)+".jpg"), image)     # save frame as JPG file
    #         return hasFrames

    #     sec = 0
    #     frameRate = 1/6 # It will capture 6 frames per second of the video 
    #     count = 1
    #     success = getFrame(sec)
    #     while success:
    #         count = count + 1
    #         sec = sec + frameRate
    #         sec = round(sec, 2)
    #         success = getFrame(sec)
    #         if count % 10 == 0:
    #             print(f'Successfully extracted {count} frames')

    #     # Crop each frames 
    #     #Extract all files from path    
    #     path = f'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\preprocess_data\/{name}'
    #     save_path = f'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master\/test_data_30\/{name}'

    #     # Get all paths from the folder -- Get only .jpg files -- Change as you want 
    #     folder_lib = pathlib.Path(path)
    #     file_names = sorted([str(item) for item in folder_lib.glob('*.jpg') if item.is_file()])

    #     # Testing
    #     for frame in file_names:
    #         image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
    #         output(image, save_path=save_path)



