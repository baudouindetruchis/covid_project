import os
import time
import requests
import cv2
from tqdm import tqdm
import urllib
import numpy as np
from matplotlib import pyplot as plt


def scrap_video(video_url, video_folder, location_name='none', frames=100, display=False):
    """Scraping from a video link"""
    cap = cv2.VideoCapture(video_url)
    # Set buffer size = 0 --> freshest frame every time
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    for i in tqdm(range(frames), desc='Recording'):
        cap = cv2.VideoCapture(video_url)
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
        cv2.imwrite(path_folder + location_name + '/' + filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

def buffer_flush(cap):
    """Empty video buffer to get low latency"""
    delay = 0
    while delay < 20:
        start = int(time.time()*1000)
        _ = cap.grab()
        delay = int(time.time()*1000) - start

# ========== RUN ==========

if __name__ == "__main__":
    # path_folder = 'D:/code#/[large_data]/covid_project/' + 'video_scraping/'
    path_folder = '/home/ec2-user/covid_project/' + 'video_scraping/'
    video_url = 'http://93.87.72.254:8090/mjpg/video.mjpg'

    scrap_video(video_url, path_folder, 'serbia', 200000, True)
