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
# Thermal

# load sample vid
try:
    h5filename = 'ThermalData_18_06_2020_13_19_36.h5'
    filename = 'E:\\GitHub\\CovPySourceFile\\' + h5filename
    file = h5py.File(filename, 'r')
    dataset = file['FRAMES']
    timestamps = file['Timestamps_ms']
except FileNotFoundError as ex:
    print(ex)
    dataset = []
    timestamps = []
except Exception as ex:
    print(ex)
    dataset = []
    timestamps = []
else:

    video = cv2.VideoCapture('E:\\GitHub\\CovPySourceFile\\Video\\' + 'OptimizedVideo.avi')
    frames = []

    while True:
        success, frame = video.read()
        if not success:
            break

        frames.append(frame)

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        predictions = fa.get_landmarks_from_image(frame)

        for p in predictions:
            if p is not None:
                for (x, y) in p:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

finally:
    print('done')
