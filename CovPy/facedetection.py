import time
import h5py
import math
import statistics
import glob
import cv2
import face_alignment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from skimage import data
from skimage.filters import threshold_multiotsu
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
from itertools import count

from cv_helper_functions import *
from hdf5_helper_functions import *
from algorithm_functions import *

import face_alignment
import collections


"""
Author: Dominik Limbach
Description:
    program processes incoming frames to extract ePPG 
"""
# Load thermal images
tag = 'NF_'
folder = 'E:\\GitHub\\CovPySourceFile\\Normalized\\'
thermal_images = load_images_from_folder(
    folder=folder,
    name_tag=tag,
)

# get timestamps
filename = 'ThermalData_18_06_2020_13_19_36.h5'
# filename = 'ThermalData_18_06_2020_13_24_58.h5'
filepath = 'E:\\GitHub\\CovPySourceFile'
dataset, timestamps = load_thermal_file(
    _filename=filename,
    _folder=filepath
)

# convert timestamps into datetime objects
dt_obj = [datetime.fromtimestamp(ts / 1000).time() for ts in timestamps]
# convert datetime objects into time strings
time_strings = [dt.strftime("%M:%S:%f") for dt in dt_obj]
# finally convert time strings into seconds
timestamp_in_seconds = []
for s in time_strings:
    date_time = datetime.strptime(s, "%M:%S:%f")
    a_timedelta = date_time - datetime(1900, 1, 1)
    in_seconds = a_timedelta.total_seconds()
    timestamp_in_seconds.append(in_seconds)

# calculate the mean interval between samples from seconds
ts_mean = np.mean(np.diff(timestamp_in_seconds))
# finally calculate the mean sampling rate of the signal
fs = int(1 / ts_mean)

# video = cv2.VideoCapture('E:\\GitHub\\CovPySourceFile\\Video\\' + 'NF_video.avi')

rectangle = [31, 35, 53, 49]
polygon = [27, 35, 53, 49, 31]
triangle = [29, 54, 48]

roi_points = []
roi_values = []
save_frames = []
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

roi_type = triangle

start_rec = 0
stop_rec = len(thermal_images)

# from image array
for ti, img in enumerate(thermal_images):
    # get landmark predictions
    if start_rec <= ti <= stop_rec:
        predictions = fa.get_landmarks_from_image(img)


        # iterate over predictions
        if predictions is not None:
            points = []
            for rp in roi_type:
                points.append([predictions[0][rp][0], predictions[0][rp][1]])

            roi_values.append(get_values_from_roi(points, img))

            for n, (x, y) in enumerate(predictions[0]):
                if n in roi_type:
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                # else:
                #   cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

            draw_roi(points, img, (0, 255, 0), 1)
            save_frames.append(img)

            roi_points.append(points)

    cv2.imshow('frame', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

# from video (works)
# while True:
#     success, frame = video.read()
#     if not success:
#         break
#
#     # get landmark predictions
#     predictions = fa.get_landmarks_from_image(frame)
#     # iterate over predictions
#     if predictions is not None:
#         for n, (x, y) in enumerate(predictions[0]):
#             if n in rectangle_points:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#             else:
#                 cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
#
#         points = []
#         for rp in rectangle_points:
#             points.append([predictions[0][rp][0], predictions[0][rp][1]])
#
#         roi_values.append(get_values_from_roi(points, frame))
#
#         draw_roi(points, frame, (0, 255, 0), 1)
#         # save_frames.append(frame)
#
#         roi_points.append(points)
#
#     cv2.imshow('frame', frame)
#
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
#
destination_dir = 'E:\\GitHub\\CovPySourceFile\\DemoVideo'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for n, f in enumerate(save_frames):
    filepath = destination_dir + '\\DF_{}.png'.format(n)
    if not os.path.exists(filepath):
        cv2.imwrite(filepath, f)

# calculate avg temp value for each frame and store it

avg_values = []

for v in roi_values:
    r_sum = np.sum(v)
    n_elements = np.count_nonzero(v)
    avg_values.append(r_sum / n_elements)

# timeline = [t / fs if t is not 0 else 0 for t in range(0, len(avg_values))]
window_timeline = [(t + start_rec) / fs if t is not 0 else start_rec for t in range(0, len(avg_values))]

plt.plot(window_timeline, avg_values)
plt.title('ROI Avg. (Raw Data)')
plt.xlabel('Time [s]')
plt.ylabel('Avg. Value')
plt.savefig('E:\\GitHub\\CovPySourceFile\\ROI_Averages.png', format='png')
plt.show()

# write values to h5 file
with h5py.File('RoiValues.h5', 'w') as f:
    f.create_dataset('ROIVALUES', data=avg_values)
    f.create_dataset('TIMELINE', data=window_timeline)

# filtering

