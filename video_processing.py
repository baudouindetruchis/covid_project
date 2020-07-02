import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm
from keras.models import load_model
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
import time

import yolo_detection
import image_cutting
import people_distances
import final_overweight_detection


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

init = 1

while True:
    # Get the last frame
    # frame = plt.imread(image_folder + sorted(os.listdir(image_folder))[-1])
    frame = plt.imread(image_folder + random.choice(os.listdir(image_folder)))
    # frame = plt.imread(image_folder + 'serbia_1593695209069.jpg')

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

    if init:
        detection_count = 0

    image_cut_list = image_cutting.cut_tolist(frame, boxes, class_ids)

    # Count detection
    detection_count = detection_count + len(image_cut_list)

    # If no detection of people continue
    if len(image_cut_list) == 0:
        continue

    # ========== OVERWEIGHT ==========

    if init:
        print("[INFO] loading overweight model from disk")
        overweight_model = load_model(project_path + 'models/overweight/' + 'overweight_detection_model_custom.h5')
        overweight_prediction = []
        overweight_perc = 0

    # Append predictions
    overweight_prediction.extend(final_overweight_detection.predict_list(image_cut_list,overweight_model))

    # Compute % overweight person per hour
    overweight_perc = round(sum(overweight_prediction)/len(overweight_prediction)*100)

    # ========== MASK ==========

    # ========== AGE ==========

    # ========== DISTANCES ==========

    if init:
        under_1m = []

    # Get distances
    under_1m.extend(people_distances.position_3d(project_path, location, boxes, class_ids, frame_height, display=False))

    # Compute % time where people are closer than 1m to another person
    under_1m_perc = round(sum(under_1m)/len(under_1m)*100)

    # ========== SAVE ==========

    print('count =', str(detection_count).ljust(5), '| under 1m % =', str(under_1m_perc).ljust(5), '| overweight % =', str(overweight_perc).ljust(5), end='\r')

    # First loop finished
    init = 0

    # Check if time to save
    now = datetime.datetime.now()
    if now.minute == 0:
        print('============================================================')
        print('[INFO] saving -', datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        print('detection count =', detection_count)
        print('under 1m % =', under_1m_perc)
        print('overweight % =', overweight_perc)
        print('============================================================')

        # Reset temp memory
        detection_count = 0
        overweight_prediction = []
        under_1m = []

        time.sleep(60)
