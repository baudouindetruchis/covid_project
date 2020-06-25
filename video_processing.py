import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm

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
    frame = plt.imread(image_folder + 'serbia_1592484700624.jpg')

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

    # If no detection no prediction
    if not boxes:
        continue

    # ========== CUTTING ==========

    image_cut_list = image_cutting.cut_tolist(frame, boxes, class_ids)

    # ========== OVERWEIGHT ==========
    if init:
        print("[INFO] loading overweight model from disk")
        with open(project_path + 'models/overweight/' + 'overweight_detection_model.pickle', 'rb') as file:
            overweight_model = pickle.load(file)

    # Save in temp folder
    for count, image in enumerate(image_cut_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(project_path + 'temp/' +'cut_' + str(count) + '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_path_list = os.listdir(project_path + 'temp/')
    image_path_list = [project_path + 'temp/' + filename for filename in image_path_list]
    print(image_path_list)

    # Predict
    overweight_prediction = final_overweight_detection.predict_list(image_path_list,overweight_model)
    print(overweight_prediction)

    # ========== MASK ==========

    # ========== AGE ==========

    # ========== DISTANCES ==========

    minimum_dist = people_distances.position_3d(project_path, location, boxes, class_ids, frame_height, display=False)
    print('minimum distance =', minimum_dist)

    # ========== SAVE ==========

    # First loop finished
    init = 0
