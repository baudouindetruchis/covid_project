import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pickle
import cv2
import cameratransform as ct
from tqdm import tqdm
import math
from itertools import combinations

import camera_calibration

# ========== FUNCTIONS ==========

def minimum_distance(topview_coor):
	# coor_2d = [[position[0], position[1]] for position in topview_coor]
	min_dist = random.randint(0,9)
	return min_dist

def position_3d(project_path, location, boxes, class_ids, frame_height, display=False):
	"""Output people topview coordinates in the real world"""
	# Get yolo prediction center
	positions = []
	for i, box in enumerate(boxes):
		if class_ids[i] == 0:
			positions.append([box[0]+box[2]//2, box[1]+box[3]//2])
	yolo_centers = np.asarray(positions)

	# Load camera parameters
	camera = ct.Camera(ct.RectilinearProjection())
	try:
		camera.load(project_path + 'camera_params/' + location + '_camera_params.json')
	except:
		camera_calibration.get_camera_params(project_path, location, display=False)
		camera.load(project_path + 'camera_params/' + location + '_camera_params.json')

	topview_coor = camera.spaceFromImage(yolo_centers, Z=0.85)                               # AVG center 1.7/2 = 0.85 m

	if display:
		plt.xlim(-5,5)
		plt.ylim(0,10)
		plt.scatter(topview_coor[:, 0], topview_coor[:, 1])
		plt.xlabel("x position in m")
		plt.ylabel("y position in m")
		plt.pause(0.2)
		plt.clf()

	min_dist = minimum_distance(topview_coor)

	return min_dist

# ========== RUN ==========

if __name__ == "__main__":
	project_path = 'D:/code#/[large_data]/covid_project/'
	location = 'serbia'
