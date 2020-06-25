# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:31:24 2020

@author: edinh
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2
from camera_calibration import process_outputs

# ========== FUNCTIONS ===========

def cutting(name,frame, boxes, labels,class_ids):
    cpt = 0
    #person
    """Take boxes in pixels with upper-left corner as reference --> draw bounding boxes on frame"""
    for i in range(len(boxes)):
        if labels[class_ids[i]] == "person":
            cpt = cpt+1
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            crop_img = frame[y:y+h, x:x+w]

            if crop_img.size != 0:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('C:/Users/edinh/Desktop/ML/covid_project-master/data/' + name + '_cut_' + str(cpt) + '.png', crop_img)

def cutting_head(frame, boxes, labels,class_ids):
    cpt = 0
    #person
    """Take boxes in pixels with upper-left corner as reference --> draw bounding boxes on frame"""
    for i in range(len(boxes)):
        if labels[class_ids[i]] == "person":
            cpt = cpt+1
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            crop_img = frame[y:y+int(h/3), x:x+w]

            if crop_img.size != 0:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('C:/Users/edinh/Desktop/ML/covid_project-master/data/' + name + '_head_cut_' + str(cpt) + '.png', crop_img)

def cut_tolist(frame, boxes, class_ids):
    """Take a frame and yolo prediction as input -->  return a list of people images"""
    image_cut_list = []
    for i in range(len(boxes)):
        # Check if detection is a person
        if class_ids[i] == 0:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            crop_img = frame[y:y+h, x:x+w]
            # Append to list
            if crop_img.size != 0:
                image_cut_list.append(crop_img)

    return image_cut_list



if __name__ == "__main__":
    # ========== YOLO SETUP ==========

    path_folder = 'C:/Users/edinh/Desktop/ML/covid_project-master/'

    # Labels & color setup
    labels = open(path_folder + 'coco.names').read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # Load model
    print("[INFO] loading YOLO from disk")
    net = cv2.dnn.readNetFromDarknet(path_folder + 'yolov3.cfg', path_folder + 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Parameters setup
    conf_threshold = 0.4	# Confidence minimum threshold
    nms_threshold = 0.4		# Non-maximum suppression threshold : overlap maximum threshold

    # ========== RUN ==========

    path_folder = 'C:/Users/edinh/Desktop/ML/covid_project-master/images'

    for element in os.listdir(path_folder):



        name = element.rsplit(".",1)[0]
        frame = plt.imread(path_folder + '/' + element)
        # frame = plt.imread(path_folder + 'hranice_1589801140_244.jpg')

        (frame_height, frame_width) = frame.shape[:2]

        # Transform frame in 416x416 blob + forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(ln)

        # Post-processing
        boxes, confidences, class_ids = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)
        cutting(name, frame, boxes, labels, class_ids)

    print('fini')                                                       # in mm
