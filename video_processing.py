import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm

import yolo_detection
# import image_cutting


# ========== PATHS ==========

project_path = 'D:/code#/[large_data]/covid_project/'
location = 'serbia'

yolo_folder = project_path + 'models/yolo_coco/'
image_folder = project_path + 'video_scraping/' + location + '/'

# ========== YOLO SETUP ==========

# Labels & color setup
labels = open(yolo_folder + 'coco.names').read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Load model
print("[INFO] loading YOLO from disk")
net = cv2.dnn.readNetFromDarknet(yolo_folder + 'yolov3.cfg', yolo_folder + 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Parameters setup
conf_threshold = 0.25	# Confidence minimum threshold
nms_threshold = 0.1		# Non-maximum suppression threshold : overlap maximum threshold


# ========== RUNING ==========


while True:
    # Get the last frame
    frame = plt.imread(image_folder + sorted(os.listdir(image_folder))[-1])

    # Transform frame in 416x416 blob + forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # Post-processing
    (frame_height, frame_width) = frame.shape[:2]
    boxes, confidences, class_ids = yolo_detection.process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)

    # Draw prediction
    yolo_frame = yolo_detection.draw_predictions(frame, boxes, confidences, class_ids, labels, colors)

    # Display the frame
    # cv2.imshow('frame', cv2.cvtColor(cv2.resize(yolo_frame, (frame_width, frame_height)), cv2.COLOR_BGR2RGB))
    # cv2.waitKey(1)

    # ========== CUTTING ==========

    # ========== OVERWEIGHT ==========

    # ========== MASK ==========

    # ========== AGE ==========

    # ========== DISTANCES ==========
