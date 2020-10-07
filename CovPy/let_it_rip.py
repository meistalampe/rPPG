import time

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from skimage.filters import unsharp_mask

from hdf5_helper_functions import *
from cv_helper_functions import *
from algorithm_functions import *


def main():

    dataset, timestamps = load_thermal_file(
        _filename='ThermalData_18_06_2020_13_19_36.h5',
        _folder='E:\\GitHub\\CovPySourceFile'
    )

    # region Control Variables
    is_writing = False
    is_drawing = False
    # endregion

    # region Data Pre-Processing

    # region Timestamps to Sampling Rate

    # # convert timestamps into datetime objects
    # dt_obj = [datetime.fromtimestamp(ts / 1000).time() for ts in timestamps]
    # # convert datetime objects into time strings
    # time_strings = [dt.strftime("%M:%S:%f") for dt in dt_obj]
    # # finally convert time strings into seconds
    # timestamp_in_seconds = []
    # for s in time_strings:
    #     date_time = datetime.strptime(s, "%M:%S:%f")
    #     a_timedelta = date_time - datetime(1900, 1, 1)
    #     in_seconds = a_timedelta.total_seconds()
    #     timestamp_in_seconds.append(in_seconds)
    #
    # # calculate the mean interval between samples from seconds
    # ts_mean = np.mean(np.diff(timestamp_in_seconds))
    # # finally calculate the mean sampling rate of the signal
    # fs = int(1 / ts_mean)
    # endregion

    # region Get Raw Thermal Data

    # get data set attributes
    n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]
    # extract thermal frames from the hdf5 dataset
    thermal_frames = []
    # convert raw data into temperature values [deg Celsius]
    # temp_frames = []
    # normalize raw data for further processing steps [0 - 255]
    norm_frames = []
    for n in range(0, n_frames):
        raw_frame = load_frame_from_dataset(dataset, height, n)
        thermal_frames.append(raw_frame)
        # temp_frames.append(raw_frame * 0.1 - 273.15)
        norm_frames.append(cv2.normalize(
            raw_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

    # get unsharpened img for edge detection later on
    unsharp_frames = []
    # for n, n_frame in enumerate(norm_frames):
    #     u_frame = unsharp_mask(image=n_frame, radius=3, amount=2)
    #     unsharp_frames.append(u_frame)
    #
    #     if is_writing:
    #         cv2.imwrite('E:\\GitHub\\CovPySourceFile\\UnsharpenedMask\\UM_{}.png'.format(n), u_frame)
    #
    #     if is_drawing:
    #         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
    #
    #         # Plotting the original image.
    #         ax[0].imshow(norm_frames[n])
    #         ax[0].set_title('Thermal Data - Normalized')
    #
    #         # ax[1].imshow(temp_frames[n])
    #         # ax[1].set_title('Temp Frame [C]')
    #
    #         ax[1].imshow(unsharp_frames[n])
    #         ax[1].set_title('Unsharpened Image')
    #
    #         # ax[1].imshow(norm_frames[n])
    #         # ax[1].set_title('Thermal Data - Normalized [0-255]')
    #
    #         plt.subplots_adjust()
    #         plt.show()
    #
    # if is_drawing:
    #     plt.close('all')

    # endregion

    # endregion

    # region Feature Extraction Algorithm

    # region Automatic ROI Detection

    # face segmentation using multi-level Otsu
    otsu_masks = multi_level_otsu(
        images=norm_frames, n_regions=4, target_region=3, method=OtsuMethods.BINARY, write=is_writing, draw=is_drawing)

    # to proceed the masks need to be converted into 3d array
    empty_array = np.zeros((height, width))
    _3d_otsu_masks = [np.dstack((mask, empty_array, empty_array)) for mask in otsu_masks]

    # use binary otsu mask to detect the face
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    # Set up tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[4]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # video = cv2.VideoCapture('E:\\GitHub\\CovPySourceFile\\Video\\OtsuMask.avi')
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

    tracked_frame = _3d_otsu_masks[0]
    # Define initial bounding box from roi
    bbox = cv2.selectROI(tracked_frame, showCrosshair=True, fromCenter=False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(tracked_frame, bbox)

    # roi points
    roi_points = []
    tracked_frames = []
    # while True:
        # # Read a new frame
        # ok, frame = video.read()
        # if not ok:
        #     break
    for mask in _3d_otsu_masks:
        tracked_frame = mask
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(tracked_frame)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2])), int(bbox[1])
            p3 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            p4 = (int(bbox[0]), int(bbox[1] + bbox[3]))
            cv2.rectangle(tracked_frame, p1, p3, (255, 0, 0), 2, 1)
            points = [p1, p2, p3, p4]
            # roi_values = get_values_from_roi(points, t_frame)
            roi_points.append(points)
        else:
            # Tracking failure
            cv2.putText(
                tracked_frame, "Tracking failure detected",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            roi_points.append([])

        # Display tracker type on frame
        cv2.putText(
            tracked_frame, tracker_type + " Tracker",
            (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(
            tracked_frame, "FPS : " + str(int(fps)),
            (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        tracked_frames.append(tracked_frame)
        # Display result
        cv2.imshow("Tracking", tracked_frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    is_writing = True
    if is_writing:
        for n, img in enumerate(tracked_frames):
            cv2.imwrite('E:\\GitHub\\CovPySourceFile\\TrackedFrames\\TF_{}.png'.format(n), img)

    norm_face_rois = []
    for n in range(0, len(roi_points)):
        # get values inside of roi
        norm_roi_values = get_values_from_roi(roi_points[n], norm_frames[n])
        # my_roi = np.zeros((roi_shapes[n][2], roi_shapes[n][3]))
        x1 = roi_points[n][0][0]
        x2 = roi_points[n][2][0]
        y1 = roi_points[n][0][1]
        y2 = roi_points[n][2][1]

        norm_face_roi = norm_roi_values[y1:y2, x1:x2]

        if is_drawing:
            cv2.imshow("ROI", norm_face_roi)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        norm_face_rois.append(norm_face_roi)

    if is_writing:
        for n, img in enumerate(tracked_frames):
            cv2.imwrite('E:\\GitHub\\CovPySourceFile\\FaceROI\\TF_{}.png'.format(n), img)
    # endregion

    # endregion

    print('Bye Bye')


if __name__ == '__main__':
    main()







