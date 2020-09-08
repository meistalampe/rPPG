import time

import cv2
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from matplotlib.animation import FuncAnimation
from itertools import count

from cv_helper_functions import *


"""
Author: Dominik Limbach
Description:
    program processes incoming frames to extract ePPG 
"""

# set font for cv text
cv_font = cv2.FONT_HERSHEY_COMPLEX
# set face detector for face-alignment
fa = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D, flip_input=False)
# set up video capture from webcam
cap = cv2.VideoCapture(0)

# set up fig for live plot

b_avg_forehead = 0
g_avg_forehead = 0
r_avg_forehead = 0

x_values = []
b_values = []
g_values = []
r_values = []

index = count()

while True:
    _, frame = cap.read()

    # extract color channels BGR
    blue_frame = frame.copy()
    # set green and red channels to 0
    blue_frame[:, :, 1] = 0
    blue_frame[:, :, 2] = 0

    green_frame = frame.copy()
    # set blue and red channels to 0
    green_frame[:, :, 0] = 0
    green_frame[:, :, 2] = 0

    red_frame = frame.copy()
    # set blue and green channels to 0
    red_frame[:, :, 0] = 0
    red_frame[:, :, 1] = 0

    # in case of channel specific presentation
    show_frame = frame
    # dimensions = show_frame.shape
    # print(dimensions)

    # get landmark predictions
    predictions = fa.get_landmarks_from_image(frame)
    # iterate over predictions
    if predictions is not None:
        for p in predictions:
            # draw landmarks
            for (x, y) in p:
                cv2.circle(show_frame, (x, y), 2, (255, 255, 255), -1)

            # create roi from landmarks
            forehead_roi = get_roi_from_landmarks_forehead(_predictions=p)
            # get pixel values from roi
            values_forehead = get_values_from_roi(_roi_points=forehead_roi, _image=show_frame)
            # draw the roi into the frame
            draw_roi(_roi_points=forehead_roi, _image=show_frame)

            # calculate the bgr avg inside of the roi
            b_avg_forehead = np.mean(values_forehead[:, 0])
            g_avg_forehead = np.mean(values_forehead[:, 1])
            r_avg_forehead = np.mean(values_forehead[:, 2])
            # print(b_avg_forehead, g_avg_forehead, r_avg_forehead)

    x_values.append(next(index))
    b_values.append(b_avg_forehead)
    g_values.append(g_avg_forehead)
    r_values.append(r_avg_forehead)

    # show image
    cv2.imshow("Frame", show_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


df = pd.DataFrame({
    'x': x_values,
    '_blue': b_values,
    '_green': g_values,
    '_red': r_values
})

plt.xlabel("sample")
plt.ylabel("value")
plt.plot('x', '_blue', data=df, color='blue')
plt.plot('x', '_green', data=df, color='green')
plt.plot('x', '_red', data=df, color='red')
plt.show()
