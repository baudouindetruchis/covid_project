# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2
import glob

# ========== YOLO SETUP ==========

path_folder = '/Users/jeanloubet/Documents/ML_Club/yolo/'

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
conf_threshold = 0.4  # Confidence minimum threshold
nms_threshold = 0.4  # Non-maximum suppression threshold : overlap maximum threshold

# ========== FUNCTION ==========
def process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold):
    """Process dnn outputs --> output selected boxes in pixels with upper-left corner as reference"""
    # Reset bounding boxes, class_ids & confidences
    boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = round(scores[class_id], 2)

            if confidence > conf_threshold:
                # Scale bounding boxes back to frame
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (center_x, center_y, width, height) = box.astype(int)

                # Upper-left corner coordinates
                x = int(center_x - (width // 2))
                y = int(center_y - (height // 2))

                boxes.append([x, y, int(width), int(height)])
                class_ids.append(class_id)
                confidences.append(round(float(confidence), 2))
    # Apply non-maxima suppression
    selected = np.array(cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                         nms_threshold)).flatten()  # NMSBoxes returns an empty tuple when no box
    for i in reversed(range(len(boxes))):
        if i not in selected:
            del boxes[i]
            del confidences[i]
            del class_ids[i]

    return boxes, confidences, class_ids
# ========== RUN ==========

path_folder = '/Users/jeanloubet/Documents/ML_Club/covid_project/recording/'
frames_pathes=glob.glob("/Users/jeanloubet/Documents/ML_Club/covid_project/recording/*.jpg")

for i in range(10):
    frame = plt.imread(frames_pathes[i])
    # frame = plt.imread(path_folder + 'hranice_1589801140_244.jpg')

    (frame_height, frame_width) = frame.shape[:2]

    # Transform frame in 416x416 blob + forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # Post-processing
    boxes, confidences, class_ids = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)
    print(boxes)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if class_ids[i] == 0:
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                #color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), [250,0,0], 2)
                text = "{}: {:.4f}".format("person", confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, [250,0,0], 2)
    # show the output image
    cv2.imshow("Image", frame)
    cv2.waitKey(0)

