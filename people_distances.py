import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm


# ========== FUNCTIONS ==========

def position_3d(boxes, class_ids, frame_height):
    """Output people position in the real world"""
    positions = []
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:
            positions.append([box[0]+box[2]//2, box[1]+box[3]//2])

    return np.asarray(positions)

# ========== RUN ==========

project_path = 'D:/code#/[large_data]/covid_project/'

yolo_folder = project_path + 'yolo_coco/'
image_folder = project_path + 'dataset_serbia_day1/'

camera = ct.Camera(ct.RectilinearProjection())
camera.load(project_path + 'camera_params.json')

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
conf_threshold = 0.4		                                                          # Confidence minimum threshold
nms_threshold = 0.7							                                          # Non-maximum suppression threshold : overlap maximum threshold

plt.figure('Topview')

for i in range(50):
    frame = plt.imread(image_folder + os.listdir(image_folder)[i])

    # Transform frame in 416x416 blob + forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # Post-processing
    (frame_height, frame_width) = frame.shape[:2]
    boxes, confidences, class_ids = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)

    # Spatial position
    centers = position_3d(boxes, class_ids, frame_height)
    topview_coor = camera.spaceFromImage(centers, Z=0.85)                               # AVG center 1.7/2 = 0.85 m

    frame = draw_predictions(frame, boxes, confidences, class_ids, labels, colors)

    # Display the frame
    cv2.imshow('frame', cv2.cvtColor(cv2.resize(frame, (frame_width, frame_height)), cv2.COLOR_BGR2RGB))
    # cv2.waitKey()

    plt.xlim(-5,5)
    plt.ylim(0,10)
    plt.scatter(topview_coor[:, 0], topview_coor[:, 1])
    plt.xlabel("x position in m")
    plt.ylabel("y position in m")
    plt.pause(0.2)
    plt.clf()
