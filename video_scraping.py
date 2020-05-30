import os
from datetime import datetime
import requests
import cv2
from tqdm import tqdm
import os
#Working directory path
dir_path = os.path.dirname(os.path.realpath(__file__))


def record_video(video_url, path_folder, location_name='none', frames=10):
    for i in tqdm(range(frames), desc='Recording'):
        # Get one frame
        response = requests.get(video_url)
        if response.status_code == 200:
            raw_image = response.content
        # Save image with timestamp
        print(datetime.utcnow().timestamp())
        timestamp = str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
        filename = location_name + '_' + timestamp +'.jpg'
        with open(path_folder + filename, 'wb') as file:
            file.write(raw_image)


# ========== RUN ==========

path_folder = dir_path + '/recording/'
video_url = 'http://85.71.106.87:60001/oneshotimage1?1589800293'

record_video(video_url, path_folder, 'hranice', 1000)
