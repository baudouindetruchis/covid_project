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
