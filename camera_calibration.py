import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


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

def feet_head_position(boxes, class_ids, frame_height):
    """Output feet & head position from bounding boxes"""
    position_feet = []
    position_head = []
    for i, box in enumerate(boxes):
        # Keep only person class & bounding boxes not touching image borders
        if class_ids[i] == 0 and box[1] > 0.05*frame_height and box[1]+box[3] < 0.95*frame_height:
            position_feet.append([box[0]+box[2]//2, box[1]+box[3]])
            position_head.append([box[0]+box[2]//2, box[1]])

    return position_feet, position_head

def yolo_collection(yolo_folder, image_folder):
	"""Run YOLOv3 algorithm to collect people feet & heads position"""
	# Load model
	# print("[INFO] loading YOLO from disk")
	net = cv2.dnn.readNetFromDarknet(yolo_folder + 'yolov3.cfg', yolo_folder + 'yolov3.weights')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# Parameters setup
	conf_threshold = 0.4															# Confidence minimum threshold
	nms_threshold = 0.7																# Non-maximum suppression threshold : overlap maximum threshold

	feet_selected = []
	heads_selected = []

	print('Collecting 100 positions: ', end='')

	for i in range(1000):
		# Load the last frame recorded
		frame = plt.imread(image_folder + sorted(os.listdir(image_folder))[-1])

		# Transform frame in 416x416 blob + forward pass
		blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		outputs = net.forward(ln)

		# Post-processing
		(frame_height, frame_width) = frame.shape[:2]
		boxes, confidences, class_ids = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)

		# Get feet & heads position
		feet_new, heads_new = feet_head_position(boxes, class_ids, frame_height)
		feet_selected.extend(feet_new)
		heads_selected.extend(heads_new)

		print('■', end='')

		# Stop calibration after 100 positions collected
		if len(feet_selected) > 100:
			break

	print('\n[INFO] collection finished')

	return np.asarray(feet_selected), np.asarray(heads_selected)

def get_camera_params(project_path, location, display=False):
	"""Compute camera parameters using yolo inputs and save it"""
	# Paths
	yolo_folder = project_path + 'models/yolo_coco/'
	image_folder = project_path + 'video_scraping/' + location + '/'

	# Get one frame for initialization
	frame = plt.imread(image_folder + sorted(os.listdir(image_folder))[-1])

	# Intrinsic camera parameters
	focal = 6.2                                                                         # in mm
	sensor_size = (6.17, 4.55)                                                          # in mm
	frame_size = (frame.shape[1], frame.shape[0])                                       # (width, height) in pixel

	# Initialize the camera
	camera = ct.Camera(ct.RectilinearProjection(focallength_mm=focal,
	                                            sensor=sensor_size,
	                                            image=frame_size))

	# Calibrate using people detection
	feet, heads = yolo_collection(yolo_folder, image_folder)
	camera.addObjectHeightInformation(feet, heads, 1.7, 0.3)							# Person average height in meter + std information

	# Fitting camera parameters
	trace = camera.metropolis([
			# ct.FitParameter("heading_deg", lower=-180, upper=180, value=0),
	        # ct.FitParameter("roll_deg", lower=-180, upper=180, value=0),
	        ct.FitParameter("elevation_m", lower=0, upper=100, value=5),
	        ct.FitParameter("tilt_deg", lower=0, upper=180, value=45)
	        ], iterations=1e4)

	# Save camera parameters
	camera.save(project_path + 'camera_params/' + location + '_camera_params.json')

	if display:
		# Display calibration information
		camera.plotTrace()
		plt.tight_layout()
		plt.show()

		plt.figure('Calibration information', figsize=(15,10))
		plt.subplot(1,2,1)
		camera.plotFitInformation(frame)
		plt.legend()

		plt.subplot(1,2,2)
		camera.getTopViewOfImage(frame, [-10, 10, 0, 20], do_plot=True)
		plt.xlabel("x position in m")
		plt.ylabel("y position in m")
		plt.show()

# ========== RUN ==========

if __name__ == "__main__":
	project_path = 'D:/code#/[large_data]/covid_project/'
	location = 'serbia'

	get_camera_params(project_path, location, display=False)
