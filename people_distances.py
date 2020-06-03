import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm


# ========== FUNCTIONS ==========

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
			confidence = round(scores[class_id],2)

			if confidence > conf_threshold:
				# Scale bounding boxes back to frame
				box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
				(center_x, center_y, width, height) = box.astype(int)

				# Upper-left corner coordinates
				x = int(center_x - (width // 2))
				y = int(center_y - (height // 2))

				boxes.append([x, y, int(width), int(height)])
				class_ids.append(class_id)
				confidences.append(round(float(confidence),2))

	# Apply non-maxima suppression
	selected = np.array(cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)).flatten()		# NMSBoxes returns an empty tuple when no box
	for i in reversed(range(len(boxes))):
		if i not in selected:
			del boxes[i]
			del confidences[i]
			del class_ids[i]

	return boxes, confidences, class_ids

def draw_predictions(frame, boxes, confidences, class_ids, labels, colors):
	"""Take boxes in pixels with upper-left corner as reference --> draw bounding boxes on frame"""
	for i in range(len(boxes)):
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		color = [int(c) for c in colors[class_ids[i]]]

		# Draw bounding box
		cv2.rectangle(frame, (x, y), (x + w, y + h), color=color, thickness=1)

		# Print label + confidence
		text = str(labels[class_ids[i]]) + ' ' + str(confidences[i])
		(text_width, text_height) = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, thickness=1)[0]
		cv2.rectangle(frame, (x, y-text_height-1), (x+text_width, y), color=color, thickness=cv2.FILLED)
		cv2.putText(frame, text, org=(x, y-1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,0,0), thickness=1)

	return frame

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
image_folder = project_path + 'dataset_hranice_day1/'

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
    cv2.imshow('frame', cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB))
    # cv2.waitKey()

    plt.xlim(-20,20)
    plt.ylim(0,50)
    plt.scatter(topview_coor[:, 0], topview_coor[:, 1])
    plt.xlabel("x position in m")
    plt.ylabel("y position in m")
    plt.pause(0.2)
    plt.clf()
