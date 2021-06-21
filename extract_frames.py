import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime

video_fname = 'babe.mp4'
img_name = 'babe'
################################################################################
#############################   EXTRACT FRAMES   ###############################
################################################################################


path = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master'
vidcap = cv2.VideoCapture(os.path.join(path, f'video_input\{video_fname}'))
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(path, f"preprocess_data/{img_name}/{img_name}"+str(count)+".jpg"), image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1/4 # It will capture 4 frames per second of the video 
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    if count % 10 == 0:
        print(f'Successfully extracted {count} frames')

