import os
import time
import requests
import cv2
from tqdm import tqdm
import urllib
import numpy as np
from matplotlib import pyplot as plt


def scrap_video(video_url, path_folder, location_name='none', frames=100, display=False):
    """Scraping from a video link"""
    cap = cv2.VideoCapture(video_url)
    for i in tqdm(range(frames), desc='Recording'):
        grabbed, frame = cap.read()
        if not grabbed:
            tqdm.write("[INFO] frame not captured")
            continue

        # Diplay live recording
        if display:
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) == 27:                                                # break on pressing 'esc'
                break

        # Filename : location + timestamp
        timestamp = str(int(time.time() * 1000))
        filename = location_name + '_' + timestamp +'.jpg'

        # Optimize (85% = size/3 : 120kb --> 40kb) + save
        cv2.imwrite(path_folder + filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

# ========== RUN ==========

path_folder = 'D:/code#/[large_data]/covid_project/' + 'scraping/'
video_url = 'http://93.87.72.254:8090/mjpg/video.mjpg'

scrap_video(video_url, path_folder, 'serbia', 2000, True)
