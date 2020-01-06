#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

# Importing standard libraries
import os
import sys
import cv2
import csv
import warnings
import numpy as np
from PIL import Image, ImageEnhance
from yolo import YOLO
from timeit import time
from pylab import array, uint8
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Importing other custom .py files
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')

# Main Function which implements the YOLOv3 Detector and DeepSort Tracking Algorithm
def main(yolo):

    # Determining the FPS of a video having variable frame rate
    # cv2.CAP_PROP_FPS is not used since it returns 'infinity' for variable frame rate videos
    filename = "clip1.mp4"
    # Determining the total duration of the video
    clip = VideoFileClip(filename)

    cap2 = cv2.VideoCapture(filename)
    co = 0
    ret2 = True
    while ret2:
        ret2, frame2 = cap2.read()
        # Determining the total number of frames
        co += 1
    cap2.release()

    # Computing the average FPS of the video
    Input_FPS = co / clip.duration

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    
    # Implementing Deep Sort algorithm
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    # Cosine distance is used as the metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    video_capture = cv2.VideoCapture(filename)

    # Define the codec and create a VideoWriter object to save the output video
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), Input_FPS, (int(video_capture.get(3)), int(video_capture.get(4))))

    # To calculate the frames processed by the deep sort algorithm per second
    fps = 0.0

    # Loop to process each frame and track people
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        t1 = time.time()

        step1 = cv2.edgePreservingFilter(frame, flags=1, sigma_s=15, sigma_r=0.1)
        step2 = cv2.detailEnhance(step1, sigma_s=40, sigma_r=0.1)
        cv2.imwrite('preprocessing.jpg', step2)

        im = Image.open("preprocessing.jpg")
        enhancer = ImageEnhance.Sharpness(im)
        enhanced_im = enhancer.enhance(6.0)
        enhanced_im.save("enhanced.jpg")

        frame = cv2.imread('enhanced.jpg')

        image = Image.fromarray(frame[...,::-1])   # BGR to RGB conversion
        boxs = yolo.detect_image(image)
        features = encoder(frame,boxs)
        
        # Getting the detections having score of 0.0 to 1.0
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression on the bounding boxes
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        head_count = 0

        # Drawing bounding box detections for people inside the store
        for det in detections:
            head_count += 1
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)

        cv2.putText(frame, str(head_count), (50, 50), 0, 1.5, (0, 255, 77), 2)

        # Write the frame onto the VideoWriter object
        out.write(frame)

        # Calculating the frames processed per second by the model  
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_count += 1
        # Printing processing status to track completion
        op = "FPS_" + str(frame_count) + "/" + str(co) + ": " + str(round(fps, 2))
        print("\r" + op , end = "")

    # Releasing objects created
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
