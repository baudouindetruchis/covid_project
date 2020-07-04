import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm
from keras.models import load_model
import datetime
import time
import tensorflow as tf
tf.get_logger().setLevel('INFO')


import yolo_detection
import image_cutting
import people_distances
import final_overweight_detection
import tunnel
import video_scraping


# ========== PATHS ==========

project_path = 'D:/code#/[large_data]/covid_project/'
# project_path = '/home/ec2-user/covid_project/'
location = 'serbia'
video_url = 'http://93.87.72.254:8090/mjpg/video.mjpg'

yolo_folder = project_path + 'models/yolo_coco/'


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
conf_threshold = 0.3	# Confidence minimum threshold
nms_threshold = 0.3		# Non-maximum suppression threshold : overlap maximum threshold


# ========== RUNNING ==========

init = 1

for i in tqdm(range(10**5), desc='Processing video', dynamic_ncols=True):       # 1.2 FPS = 100.000 frames /day

    # ========== INITIALIZATION ==========

    if init:
        # Recording
        cap = cv2.VideoCapture(video_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Cutting
        detection_count = 0

        # Overweight
        tqdm.write("[INFO] loading overweight model from disk")
        overweight_model = load_model(project_path + 'models/overweight/' + 'overweight_detection_model_custom.h5')
        overweight_prediction = []
        fitness_score = 0

        # Mask
        mask_compliance = 'NULL'

        # Age
        avg_age = 'NULL'

        # Gender
        gender = 'NULL'

        # Distances
        under_1m = []
        under_1m_perc = 0

        # Initialization completed
        init = 0

    # ========== RECORDING ==========

    # Flush video buffer
    video_scraping.buffer_flush(cap)

    # Get the last frame
    grabbed, frame = cap.read()
    if not grabbed:
        tqdm.write("[DEBUG] frame not captured")

    # Check if frame
    elif frame.ndim != 3:
        tqdm.write("[DEBUG] empty frame")

    else:

        # Get RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ========== YOLO RUN ==========

        # Transform frame in 416x416 blob + forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=False, crop=False)
        net.setInput(blob)
        outputs = net.forward(ln)

        # Post-processing
        (frame_height, frame_width) = frame.shape[:2]
        boxes, confidences, class_ids = yolo_detection.process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)

        # Draw prediction
        yolo_frame = yolo_detection.draw_predictions(frame, boxes, confidences, class_ids, labels, colors)

        # Display the frame
        # cv2.imshow('frame', cv2.cvtColor(cv2.resize(yolo_frame, (frame_width, frame_height)), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # ========== CUTTING ==========

        image_cut_list = image_cutting.cut_tolist(frame, boxes, class_ids)

        # Count detection
        detection_count = detection_count + len(image_cut_list)

        # If detection run models
        if len(image_cut_list) != 0:

            # ========== OVERWEIGHT ==========

            # Append predictions
            overweight_prediction.extend(final_overweight_detection.predict_list(image_cut_list,overweight_model))

            # Compute fitness score
            fitness_score = 100 - round(sum(overweight_prediction)/len(overweight_prediction)*100)

            # ========== MASK ==========

            # ========== AGE ==========

            # ========== GENDER ==========

            # ========== DISTANCES ==========

            # Get distances
            under_1m.extend(people_distances.position_3d(project_path, location, boxes, class_ids, frame_height, display=False))

            # Compute % time where people are closer than 1m to another person
            under_1m_perc = round(sum(under_1m)/len(under_1m)*100)

    # ========== SAVE ==========

    # Check if time to save
    now = datetime.datetime.now()
    if now.second == 0 or now.second == 1 or now.second == 2 or now.second == 3:

        # Print report
        tqdm.write('[INFO] saving - time = ' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        tqdm.write('count: ' + str(detection_count).ljust(5) + ' | <1m: ' + str(under_1m_perc).rjust(2) + '% | '
                    + 'mask: ' + str(mask_compliance).rjust(2) + '% | gender: ' + str(gender) + ' | '
                    + 'avg_age: ' + str(avg_age).rjust(2) + ' | fit_score: ' + str(fitness_score).rjust(2) + '%')

        # Send to database
        try:
            tunnel.insertLog("Serbia2", now.strftime('%Y%m%d%H%M%S'), detection_count, mask_compliance, under_1m_perc, gender, avg_age, fitness_score)
        except:
            tqdm.write('[WARNING] cannot save to DB')

        # Reset temp memory
        detection_count = 0
        overweight_prediction = []
        fitness_score = 'NULL'
        mask_compliance = 'NULL'
        avg_age = 'NULL'
        gender = 'NULL'
        under_1m = []
        under_1m_perc = 0

        # Wait the end of condition
        time.sleep(4)
