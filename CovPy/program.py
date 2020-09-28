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
from datetime import datetime, timedelta

import scipy
from scipy import signal
from scipy import fftpack
from matplotlib.animation import FuncAnimation
from itertools import count

from scipy.signal import tf2zpk

from cv_helper_functions import *
from hdf5_helper_functions import *
from algorithm_functions import *


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
    h5file = h5py.File(filename, 'r')
    dataset = h5file['FRAMES']
    timestamps = h5file['Timestamps_ms']
except FileNotFoundError as ex:
    print(ex)
    dataset = []
    timestamps = []
except Exception as ex:
    print(ex)
    dataset = []
    timestamps = []
else:
    matplotlib.rcParams['font.size'] = 9
    dtime = [datetime.fromtimestamp(ts/1000).time() for ts in timestamps]
    time_strs = [dt.strftime("%M:%S:%f") for dt in dtime]
    time_in_seconds = []
    for s in time_strs:
        date_time = datetime.strptime(s, "%M:%S:%f")
        a_timedelta = date_time - datetime(1900, 1, 1)
        seconds = a_timedelta.total_seconds()
        time_in_seconds.append(seconds)

    mean_fps = np.mean(np.diff(time_in_seconds))
    fs = int(1/mean_fps)
    # --- Multi Otsu to Face ROI --- #

    # n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]
    #
    # # get roi points from multi level otsu
    # (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    #
    # # Set up tracker
    # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
    #                  'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[4]
    #
    # if int(minor_ver) < 3:
    #     tracker = cv2.Tracker_create(tracker_type)
    # else:
    #     if tracker_type == 'BOOSTING':
    #         tracker = cv2.TrackerBoosting_create()
    #     if tracker_type == 'MIL':
    #         tracker = cv2.TrackerMIL_create()
    #     if tracker_type == 'KCF':
    #         tracker = cv2.TrackerKCF_create()
    #     if tracker_type == 'TLD':
    #         tracker = cv2.TrackerTLD_create()
    #     if tracker_type == 'MEDIANFLOW':
    #         tracker = cv2.TrackerMedianFlow_create()
    #     if tracker_type == 'GOTURN':
    #         tracker = cv2.TrackerGOTURN_create()
    #     if tracker_type == 'MOSSE':
    #         tracker = cv2.TrackerMOSSE_create()
    #     if tracker_type == "CSRT":
    #         tracker = cv2.TrackerCSRT_create()
    #
    # # Read video
    # video_frames = []
    # video = cv2.VideoCapture('E:\\GitHub\\CovPySourceFile\\Video\\' + 'OtsuMaskVideo.avi')
    # # images = load_images_from_folder('E:\\GitHub\\CovPySourceFile\\TrackingMask')
    # # Exit if video not opened.
    # if not video.isOpened():
    #     print("Could not open video file!")
    #     sys.exit()
    #
    # # Read first frame
    # ok, frame = video.read()
    # if not ok:
    #     print("Could not read video file!")
    #     sys.exit()
    #
    # # Define initial bounding box from roi
    # bbox = cv2.selectROI(frame, showCrosshair=True, fromCenter=False)
    #
    # # Initialize tracker with first frame and bounding box
    # ok = tracker.init(frame, bbox)
    #
    # # roi points
    # roi_points = []
    # roi_shapes = []
    # while True:
    #     # Read a new frame
    #     ok, frame = video.read()
    #     if not ok:
    #         break
    #     # Start timer
    #     timer = cv2.getTickCount()
    #
    #     # Update tracker
    #     ok, bbox = tracker.update(frame)
    #     # Calculate Frames per second (FPS)
    #     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #
    #     # Draw bounding box
    #     if ok:
    #         # Tracking success
    #         p1 = (int(bbox[0]), int(bbox[1]))
    #         p2 = (int(bbox[0] + bbox[2])), int(bbox[1])
    #         p3 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #         p4 = (int(bbox[0]), int(bbox[1] + bbox[3]))
    #         cv2.rectangle(frame, p1, p3, (255, 0, 0), 2, 1)
    #         # cv2.circle(frame, p1, 4, (255, 0, 0), -1)
    #         # cv2.circle(frame, p2, 4, (255, 0, 0), -1)
    #         # cv2.circle(frame, p3, 4, (0, 0, 255), -1)
    #         # cv2.circle(frame, p4, 4, (0, 255, 0), -1)
    #         points = [p1, p2, p3, p4]
    #         # roi_values = get_values_from_roi(points, t_frame)
    #         roi_points.append(points)
    #     else:
    #         # Tracking failure
    #         cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    #         roi_points.append([])
    #
    #     # Display tracker type on frame
    #     cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    #
    #     # Display FPS on frame
    #     cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    #
    #     video_frames.append(frame)
    #     # Display result
    #     cv2.imshow("Tracking", frame)
    #
    #     # Exit if ESC pressed
    #     k = cv2.waitKey(1) & 0xff
    #     if k == 27:
    #         break
    #
    # print('Roi_points: {}'.format(len(roi_points)))
    # print('Thermal_frames: {}'.format(n_frames))
    #
    # destination_dir = 'E:\\GitHub\\CovPySourceFile\\FaceROI\\'
    # th_filename = 'ThFaceROI_'
    # norm_filename = 'NormFaceROI'
    #
    # # cv2.destroyAllWindows()
    # for n in range(0, len(roi_points)):
    #     th_frame = load_frame_from_dataset(dataset=dataset, frame_height=height, frame_number=n)
    #     temp_frame = th_frame * 0.1 - 273.15
    #
    #     norm_frame = cv2.normalize(temp_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    #     # get values inside of roi
    #     th_roi_values = get_values_from_roi(roi_points[n], temp_frame)
    #     norm_roi_values = get_values_from_roi(roi_points[n], norm_frame)
    #     # my_roi = np.zeros((roi_shapes[n][2], roi_shapes[n][3]))
    #     x1 = roi_points[n][0][0]
    #     x2 = roi_points[n][2][0]
    #     y1 = roi_points[n][0][1]
    #     y2 = roi_points[n][2][1]
    #
    #     th_face_roi = th_roi_values[y1:y2, x1:x2]
    #     norm_face_roi = norm_roi_values[y1:y2, x1:x2]
    #     cv2.imwrite(destination_dir + th_filename + '{}.png'.format(n), th_face_roi)
    #     cv2.imwrite(destination_dir + norm_filename + '{}.png'.format(n), norm_face_roi)
    #     cv2.imshow("ROI", norm_face_roi)
    #
    #     # Exit if ESC pressed
    #     k = cv2.waitKey(1) & 0xff
    #     if k == 27:
    #         break

    # --- Face ROI to temp --- #
    # file_tag = 'ThFace'
    # face_images = load_images_from_folder('E:\\GitHub\\CovPySourceFile\\FaceROI', file_tag=file_tag)
    #
    # face_mean_temp = []
    # for img in face_images:
    #     face_mean_temp.append(np.mean(img))
    #
    # modes = ['full', 'same', 'valid']
    # window = 50
    # plt.figure()
    # face_ma_temp = np.convolve(face_mean_temp, np.ones((window,))/window, mode=modes[2])
    # plt.plot(face_ma_temp)
    # plt.show()

    image_dir = 'E:\\GitHub\\CovPySourceFile\\FaceROI\\'
    file_type = '.png'
    image_name_tag = 'NormFaceROI'
    # face_images = load_images_from_folder('E:\\GitHub\\CovPySourceFile\\FaceROI', file_tag=file_tag)

    n_images = len([file for file in os.listdir(image_dir) if file.endswith(file_type) and image_name_tag in file])
    face_images = []
    search_windows = []
    temp = []
    for n in range(0, n_images):
        img_name = image_dir + image_name_tag + '{}'.format(n) + file_type
        img = cv2.imread(img_name)
        face_images.append(img)
        print(n)

    window = 500
    for img in face_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        # x = y i.e. |
        x_min = int(size[0]*0.4)
        x_max = int(size[0]*0.7)
        # y = x i.e. __
        y_min = int(size[1]*0.3)
        y_max = int(size[1]*0.7)

        search_window = gray[x_min:x_max+1, y_min:y_max+1]
        search_windows.append(search_window)
        max_value = np.amax(search_window)
        show_me_eyes = np.where(search_window >= max_value * 0.95, search_window, 0)

        flat_arr = show_me_eyes.flatten()
        flat_non_zero = flat_arr[flat_arr != 0]
        mean_temp = np.mean(flat_non_zero)
        # print('max: {}, mean: {}'.format(max_value, mean_temp))
        # cv2.imshow('eyes', show_me_eyes)
        # cv2.imshow('ROM', search_window)

        temp.append(mean_temp)
        ma_temp = np.convolve(temp, np.ones((window,)) / window, mode='valid')

        # k = cv2.waitKey(1)
        # if k == 27:
        #     break

    cv2.destroyAllWindows()

    plt.plot(ma_temp)
    plt.show()

    mag_map = gradiant_mapping(search_windows, kernel_type=KernelMethods.CANNY)

    horizontal_borders = []
    vertical_center = []
    for m in mag_map[:3000]:
        vert_proj = cv2.reduce(m, 1, cv2.REDUCE_SUM, None, dtype=cv2.CV_64F)
        hori_proj = cv2.reduce(m, 0, cv2.REDUCE_SUM, None, dtype=cv2.CV_64F)
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

        # # Plotting the original image.
        # ax[0].imshow(m)
        # ax[0].set_title('Original')
        # ax[0].axis('off')

        # horizontal projection
        # find first and last value != 0
        h_trans = hori_proj.transpose()
        from_left = np.where(h_trans > 700)
        # print(first[0][0])
        from_right = np.where(np.flip(h_trans) > 700)
        if len(from_left[0]) == 0:
            left_border = 0
        elif len(from_right[0]) == 0:
            right_border = 0
        else:
            left_border = from_left[0][0]
            right_border = from_right[0][0]

        horizontal_borders.append((left_border, right_border))
        # print(left_border, right_border)
        # print(last[0][0])
        # ax[1].plot(h_trans)
        # ax[1].plot(first[0][0], h_trans[first[0][0]], 'b*')
        # ax[1].plot(len(h_trans) - last[0][0] - 1, np.flip(h_trans)[last[0][0]], 'b*')
        # ax[1].set_title('Horizontal Projection')

        # vertical projection
        # ax[2].plot(vert_proj)
        # ax[2].plot(vert_proj.argmax(), np.amax(vert_proj), 'b*')
        # ax[2].set_title('Vertical Projection')
        vertical_center.append(vert_proj.argmax())
        # plt.subplots_adjust()
        #
        # plt.show()

        k = cv2.waitKey(1)
        if k == 27:
            break

    roms = []
    temp_rom = []
    for n, img in enumerate(search_windows[:3000]):
        vert_window_scale = 0.15
        p1 = (int(horizontal_borders[n][0]), int(vertical_center[n] - vert_window_scale * img.shape[0]))
        p2 = (img.shape[1]-int(horizontal_borders[n][1]), int(vertical_center[n] + vert_window_scale * img.shape[0]))
        # cv2.rectangle(img, p1, p2, (255, 0, 0), 1)
        # cv2.line(img, (0, int(vertical_center[n] + vert_window_scale * img.shape[0])), (img.shape[1], int(vertical_center[n] + 0.1 * img.shape[0])), (0, 0, 255), 1)
        # cv2.line(img, (0, int(vertical_center[n] - vert_window_scale * img.shape[0])), (img.shape[1], int(vertical_center[n] - 0.1 * img.shape[0])), (0, 0, 255), 1)
        # cv2.line(img, (int(horizontal_borders[n][0]), 0), (int(horizontal_borders[n][0]), img.shape[0]), (0, 255, 255), 1)
        # cv2.line(img, (img.shape[1]-int(horizontal_borders[n][1]), 0), (img.shape[1]-int(horizontal_borders[n][1]), img.shape[0]), (0, 255, 255), 1)
        rom = img[int(vertical_center[n] - vert_window_scale * img.shape[0]):int(vertical_center[n] + vert_window_scale * img.shape[0]) + 1, int(horizontal_borders[n][0]):img.shape[1]-int(horizontal_borders[n][1]) + 1]
        roms.append(rom)
        # cv2.imshow('ROM', img)
        # plt.imshow(rom)
        #  plt.show()
        # k = cv2.waitKey(1)
        # if k == 27:
        #     break
        # cv2.imwrite('E:\\GitHub\\CovPySourceFile\\Rom\\Rom_{}.png'.format(n), rom)

        # --- detect breathing rate --- #
        # get mean temp
        if rom.size != 0:
            temp_rom.append(np.mean(rom))
        else:
            temp_rom.append(0)

    data = np.array(temp_rom)
    N = len(data)

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y


    lowcut = 0.1
    highcut = 0.85
    b, a = butter_bandpass(lowcut, highcut, fs, order=2)
    temp_filtered = butter_bandpass_filter(temp_rom, lowcut, highcut, fs, order=2)
    # y = scipy.fft.fft(temp_filtered)
    # yf = 2.0/N * np.abs(y[0:N//2])
    # T = 1.0 / len(temp_filtered)
    # xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    # br = fs * np.where(yf == np.amax(yf))[0]#

    y = fftpack.fft(temp_filtered)
    freqs = fftpack.fftfreq(len(temp_filtered)) * fs

    fr = freqs[0:int(fs/2)]
    br = fr[np.where(y == np.amax(y[0:int(fs/2)]))[0]]
    print(br)
    # fig, ax = plt.subplots()
    #
    # ax.stem(fr, np.abs(y[0:int(fs/2)]))
    # ax.set_xlabel('Frequency in Hertz [Hz]')
    # ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    # ax.set_xlim(0, fs / 2)
    breaths_per_min = []

    temp_chunks = []
    window_size = 100
    for i in range(0, len(temp_filtered), window_size):
        data = temp_filtered[i:i + window_size]
        window = signal.hamming(window_size, sym=0)
        data *= window
        # FFT transform and modulus squared
        fft = numpy.fft.fft(data)
        fft = numpy.absolute(fft)
        fft = numpy.square(fft)
        # Frequency samples
        frequencies = numpy.fft.fftfreq(
            data.size,
            1/fs
        )
        # Find the index of the maximum FFT value ,
        # and get the respiration frequency
        max_idx = np.argmax(fft)
        breaths_per_sec = frequencies[max_idx]
        breaths_per_min.append(breaths_per_sec * 60)


    #
    # print(temp_chunks)
    #
    # br_per_sec = []
    # secs = []
    # for n, c in enumerate(temp_chunks):
    #     y = fftpack.fft(c)
    #     freqs = fftpack.fftfreq(len(c)) * fs
    #
    #     fr = freqs[0:int(fs / 2)]
    #     br = fr[np.where(y == np.amax(y[0:int(fs / 2)]))[0]]
    #
    #     br_per_sec.append(6*br)
    #     secs.append(n)
    ##
    ma_bpm = np.convolve(breaths_per_min, np.ones((5,)) / 5, mode='valid')
    plt.plot(ma_bpm)
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
    #
    # ax[0].plot(temp_rom)
    # ax[0].set_title('ROM temp')
    # ax[1].plot(temp_filtered)
    # ax[1].set_title('Filtered temp')

finally:
    print('done')


# (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
#
# # Set up tracker
# tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
#                  'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# tracker_type = tracker_types[4]
#
# if int(minor_ver) < 3:
#     tracker = cv2.Tracker_create(tracker_type)
# else:
#     if tracker_type == 'BOOSTING':
#         tracker = cv2.TrackerBoosting_create()
#     if tracker_type == 'MIL':
#         tracker = cv2.TrackerMIL_create()
#     if tracker_type == 'KCF':
#         tracker = cv2.TrackerKCF_create()
#     if tracker_type == 'TLD':
#         tracker = cv2.TrackerTLD_create()
#     if tracker_type == 'MEDIANFLOW':
#         tracker = cv2.TrackerMedianFlow_create()
#     if tracker_type == 'GOTURN':
#         tracker = cv2.TrackerGOTURN_create()
#     if tracker_type == 'MOSSE':
#         tracker = cv2.TrackerMOSSE_create()
#     if tracker_type == "CSRT":
#         tracker = cv2.TrackerCSRT_create()
#
# # Read video
# video_name = ['ThermalVideo.avi', 'OptimizedVideo.avi', 'MagnitudeMap_Laplacian_k5g9.avi', 'OtsuMaskVideo.avi']
# video = cv2.VideoCapture('E:\\GitHub\\CovPySourceFile\\Video\\' + video_name[3])
#
# # Exit if video not opened.
# if not video.isOpened():
#     print("Could not open video file!")
#     sys.exit()
#
# # Read first frame
# ok, frame = video.read()
# if not ok:
#     print("Could not read video file!")
#     sys.exit()
#
# # Define initial bounding box from roi
# bbox = cv2.selectROI(frame, showCrosshair=True, fromCenter=False)
#
# # Initialize tracker with first frame and bounding box
# ok = tracker.init(frame, bbox)
#
# while True:
#     # Read a new frame
#     ok, frame = video.read()
#     if not ok:
#         break
#
#     # Start timer
#     timer = cv2.getTickCount()
#
#     # Update tracker
#     ok, bbox = tracker.update(frame)
#     print(bbox)
#     # Calculate Frames per second (FPS)
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#
#     # Draw bounding box
#     if ok:
#         # Tracking success
#         p1 = (int(bbox[0]), int(bbox[1]))
#         p2 = (int(bbox[0] + bbox[2])), int(bbox[1])
#         p3 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#         p4 = (int(bbox[0]), int(bbox[1] + bbox[3]))
#         # cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
#         cv2.circle(frame, p1, 4, (255, 0, 0), -1)
#         cv2.circle(frame, p2, 4, (255, 0, 0), -1)
#         cv2.circle(frame, p3, 4, (0, 0, 255), -1)
#         cv2.circle(frame, p4, 4, (0, 255, 0), -1)
#         points = [p1, p2, p3, p4]
#         roi_values = get_values_from_roi(points, frame)
#     else:
#         # Tracking failure
#         cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#
#     # Display tracker type on frame
#     cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
#
#     # Display FPS on frame
#     cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
#
#     # get pixel values from roi
#     # roi_points = [(int(bbox[0]), )]
#     # values_roi = get_values_from_roi(_roi_points=forehead_roi, _image=frame)
#
#     # Display result
#     # print(bbox, bbox[1])
#     cv2.imshow("Tracking", roi_values)
#
#     # Exit if ESC pressed
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()









    # n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]
    #
    # frame_number = 400
    # # load frame
    # if 0 <= frame_number <= n_frames:
    #     frame = load_frame_from_dataset(dataset=dataset, frame_height=height, frame_number=frame_number)
    # else:
    #     print("Index out of bounds! Could not load frame number {}".format(frame_number))
    #     frame = []
    #
    # # the following needs to be done for every frame
    # # frame = frame / 10 - 273.15
    # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # # Optimal Quantization for high thermal dynamic range scenes
    # f_mean = np.mean(frame)
    # f_std = np.std(frame)
    # t_min = f_mean - 1.96 * (f_std / math.sqrt(height * width))
    # t_max = f_mean + 1.96 * (f_std / math.sqrt(height * width))
    # # Ridler and Calvard’s concept of optimal threshold selection
    # # initialize the optimal temperature threshold as t_min
    # t_opt = t_min
    # # iterate until t_threshold_optimal(i) - t_threshold_optimal(i-1) ~ 0
    # while True:
    #     # thresholding_array = np.zeros((height, width))
    #     t_opt_old = t_opt
    #     mu_lower = np.mean(frame[frame < t_opt])
    #     mu_higher = np.mean(frame[frame > t_opt])
    #     t_opt = 0.5 * (mu_lower + mu_higher)
    #     print("T_opt(p): {}    T_opt(p-1): {}".format(t_opt, t_opt_old))
    #
    #     if math.isclose(t_opt, t_opt_old, rel_tol=1e-5):
    #         break
    #
    # # hist_full = cv2.calcHist([frame], [0], None, [256], [0, 256])
    # # plt.plot(hist_full)
    # # plt.show()
    # t_range = (t_opt, t_max)
    # print(t_range)
    # # my_img = np.where(frame <= t_opt, 255, 0)
    # # plt.imshow(my_img)
    # # plt.show()
    #
    # img_normalized = cv2.normalize(frame, None, alpha=t_range[0], beta=t_range[1], norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # # img_normalized = cv2.applyColorMap(img_normalized, cv2.COLORMAP_HOT)
    #
    # while True:
    #     cv2.imshow('bf', img_normalized)
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    #
    # print()

    # while True:
    #
    #     frame = load_frame_from_dataset(dataset, height, 300)
    #     raw_img = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    #     cv2.imshow('image', raw_img)
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break



# RGB
# # set font for cv text
# cv_font = cv2.FONT_HERSHEY_COMPLEX
# # set face detector for face-alignment
# fa = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D, flip_input=False)
# # set up video capture from webcam
# cap = cv2.VideoCapture(0)
#
# # set up fig for live plot
#
# b_avg_forehead = 0
# g_avg_forehead = 0
# r_avg_forehead = 0
#
# x_values = []
# b_values = []
# g_values = []
# r_values = []
#
# index = count()
#
# while True:
#     _, frame = cap.read()
#
#     # extract color channels BGR
#     blue_frame = frame.copy()
#     # set green and red channels to 0
#     blue_frame[:, :, 1] = 0
#     blue_frame[:, :, 2] = 0
#
#     green_frame = frame.copy()
#     # set blue and red channels to 0
#     green_frame[:, :, 0] = 0
#     green_frame[:, :, 2] = 0
#
#     red_frame = frame.copy()
#     # set blue and green channels to 0
#     red_frame[:, :, 0] = 0
#     red_frame[:, :, 1] = 0
#
#     # in case of channel specific presentation
#     show_frame = frame
#     # dimensions = show_frame.shape
#     # print(dimensions)
#
#     # get landmark predictions
#     predictions = fa.get_landmarks_from_image(frame)
#     # iterate over predictions
#     if predictions is not None:
#         for p in predictions:
#             # draw landmarks
#             for (x, y) in p:
#                 cv2.circle(show_frame, (x, y), 2, (255, 255, 255), -1)
#
#             # create roi from landmarks
#             forehead_roi = get_roi_from_landmarks_forehead(_predictions=p)
#             # get pixel values from roi
#             values_forehead = get_values_from_roi(_roi_points=forehead_roi, _image=show_frame)
#             # draw the roi into the frame
#             draw_roi(_roi_points=forehead_roi, _image=show_frame)
#
#             # calculate the bgr avg inside of the roi
#             b_avg_forehead = np.mean(values_forehead[:, 0])
#             g_avg_forehead = np.mean(values_forehead[:, 1])
#             r_avg_forehead = np.mean(values_forehead[:, 2])
#             # print(b_avg_forehead, g_avg_forehead, r_avg_forehead)
#
#     x_values.append(next(index))
#     b_values.append(b_avg_forehead)
#     g_values.append(g_avg_forehead)
#     r_values.append(r_avg_forehead)
#
#     # show image
#     cv2.imshow("Frame", show_frame)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
#
# df = pd.DataFrame({
#     'x': x_values,
#     '_blue': b_values,
#     '_green': g_values,
#     '_red': r_values
# })
#
# plt.xlabel("sample")
# plt.ylabel("value")
# plt.plot('x', '_blue', data=df, color='blue')
# plt.plot('x', '_green', data=df, color='green')
# plt.plot('x', '_red', data=df, color='red')
# plt.show()
