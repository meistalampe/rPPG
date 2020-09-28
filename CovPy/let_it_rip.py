import time

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from hdf5_helper_functions import *
from cv_helper_functions import *
from algorithm_functions import *


def main():
    try:
        # load hdf5 file
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
        # region Control Variables

        is_writing = False
        is_drawing = False

        # endregion

        # region Data Pre-Processing

        # region Timestamps to Sampling Rate

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
        # endregion

        # region Get Raw Thermal Data

        # get data set attributes
        n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]
        # extract thermal frames from the hdf5 dataset
        thermal_frames = []
        # convert raw data into temperature values [deg Celsius]
        temp_frames = []
        # normalize raw data for further processing steps [0 - 255]
        norm_frames = []
        for n in range(0, n_frames):
            raw_frame = load_frame_from_dataset(dataset, height, n)
            thermal_frames.append(raw_frame)
            temp_frames.append(raw_frame * 0.1 - 273.15)
            norm_frames.append(cv2.normalize(raw_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8U))

            if is_drawing:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

                # Plotting the original image.
                ax[0].imshow(raw_frame)
                ax[0].set_title('Thermal Data - Raw')

                ax[1].imshow(temp_frames[n])
                ax[1].set_title('Temp Frame [C]')

                ax[2].imshow(norm_frames[n])
                ax[2].set_title('Thermal Data - Normalized [0-255]')

                plt.subplots_adjust()
                plt.show()

        if is_drawing:
            plt.close('all')

        # endregion

        # endregion

        # region Feature Extraction Algorithm

        # region Automatic ROI Detection

        # face segmentation using multi-level Otsu
        def multi_level_otsu(thermal_data, n_regions: int = 4, target_region: int = 3, write: bool = False,
                             _destination_dir: str = 'E:\\GitHub\\CovPySourceFile\\MultiLevelOtsu\\',
                             file_name: str = 'MLO_', draw: bool = False):
            """
            Function applies multi-level Otsu algorithm to seperate the image background from the foreground.

            :param thermal_data:
                input the normalized thermal data.
            :param n_regions:
                number of layers the algorithm tries to create.
            :param target_region:
                number of the closest layer. This is the foreground and should contain the desired face regions.
            :param write:
                when true the function will write the resulting images to .png files.
            :param _destination_dir:
                save path for image files.
            :param file_name:
                name tag for the saved files.
            :param draw
                when true the generated images are plotted.
            :return:
                a list of images. Each image is a otsu mask for the target region of the corresponding thermal frame.
            """
            otsu_images = []

            # apply multi-level Otsu threshold for the input value n_regions, generating just as many classes
            for th_n, frame in enumerate(thermal_data):
                start = time.time()

                thresholds = threshold_multiotsu(image=frame, classes=n_regions)
                # use the threshold values to generate the classes
                regions = np.digitize(frame, bins=thresholds)

                # check for region sizes
                # sort regions array
                # sorted_regions = np.sort(regions, axis=None)
                # diff_regions = np.diff(sorted_regions)
                # region_shifts = np.where(diff_regions != 0)
                # region_sizes = []
                # for r in range(0, len(region_shifts[0])):
                #     if r == 0:
                #         region_sizes.append(region_shifts[0][r])
                #     else:
                #         region_sizes.append(region_shifts[0][r] - region_shifts[0][r - 1])
                #
                # region_sizes.append(len(sorted_regions) - region_shifts[0][-1])
                # print(region_sizes)

                # extract the target region

                # new solution
                otsu_mask = np.where(regions == target_region, frame, 0)
                otsu_images.append(otsu_mask)
                # old and slow solution
                # otsu_mask = np.zeros((height, width))
                # for r in range(height):
                #     for c in range(width):
                #         if regions[r, c] == target_region:
                #             otsu_mask[r, c] = 1

                # prim_region = np.where(otsu_mask == 1, frame, 0)
                # otsu_images.append(prim_region)
                end = time.time()
                print('MLO: Frame processed in {}'.format(end-start))

                if draw:
                    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 3.5))

                    # plotting the original image.
                    ax[0][0].imshow(frame, cmap='gray')
                    ax[0][0].set_title('Original')

                    # plotting the target region.
                    ax[0][1].imshow(otsu_mask)
                    ax[0][1].set_title('Target Region')

                    # plotting the histogram and the two thresholds obtained from multi-Otsu.
                    ax[1][0].hist(frame.ravel(), bins=255)
                    ax[1][0].set_title('Histogram')
                    for thresh in thresholds:
                        ax[1][0].axvline(thresh, color='r')

                    # Plotting the Multi Otsu result.
                    ax[1][1].imshow(regions, cmap='Accent')
                    ax[1][1].set_title('Multi-Otsu result')

                    plt.subplots_adjust()
                    plt.show()

                if write:
                    cv2.imwrite(_destination_dir + file_name + '{}.png'.format(th_n), otsu_mask)

            if draw:
                plt.close('all')

            return otsu_images

        # is_drawing = True
        # is_writing = True
        otsu_masks = multi_level_otsu(thermal_data=norm_frames, n_regions=4, target_region=3, write=is_writing,
                                      draw=is_drawing)

        # to proceed the masks need to be converted into 3d array
        _3d_otsu_masks = [np.dstack((mask, mask, mask)) for mask in otsu_masks]

        # use binary otsu mask to detect the face
        (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

        # Set up tracker
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                         'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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

        frame = _3d_otsu_masks[0]
        # Define initial bounding box from roi
        bbox = cv2.selectROI(frame, showCrosshair=True, fromCenter=False)

        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

        # roi points
        roi_points = []

        # while True:
            # # Read a new frame
            # ok, frame = video.read()
            # if not ok:
            #     break
        for mask in _3d_otsu_masks:
            frame = mask
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2])), int(bbox[1])
                p3 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                p4 = (int(bbox[0]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p3, (255, 0, 0), 2, 1)
                # cv2.circle(frame, p1, 4, (255, 0, 0), -1)
                # cv2.circle(frame, p2, 4, (255, 0, 0), -1)
                # cv2.circle(frame, p3, 4, (0, 0, 255), -1)
                # cv2.circle(frame, p4, 4, (0, 255, 0), -1)
                points = [p1, p2, p3, p4]
                # roi_values = get_values_from_roi(points, t_frame)
                roi_points.append(points)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                roi_points.append([])

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        destination_dir = 'E:\\GitHub\\CovPySourceFile\\FaceROI\\'
        th_filename = 'ThFaceROI_'
        norm_filename = 'NormFaceROI'

        # cv2.destroyAllWindows()
        for n in range(0, len(roi_points)):
            th_frame = load_frame_from_dataset(dataset=dataset, frame_height=height, frame_number=n)
            temp_frame = th_frame * 0.1 - 273.15

            norm_frame = cv2.normalize(temp_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # get values inside of roi
            th_roi_values = get_values_from_roi(roi_points[n], temp_frame)
            norm_roi_values = get_values_from_roi(roi_points[n], norm_frame)
            # my_roi = np.zeros((roi_shapes[n][2], roi_shapes[n][3]))
            x1 = roi_points[n][0][0]
            x2 = roi_points[n][2][0]
            y1 = roi_points[n][0][1]
            y2 = roi_points[n][2][1]

            th_face_roi = th_roi_values[y1:y2, x1:x2]
            norm_face_roi = norm_roi_values[y1:y2, x1:x2]
            cv2.imwrite(destination_dir + th_filename + '{}.png'.format(n), th_face_roi)
            cv2.imwrite(destination_dir + norm_filename + '{}.png'.format(n), norm_face_roi)
            cv2.imshow("ROI", norm_face_roi)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        # endregion

        # endregion
    finally:
        print('Bye Bye')


if __name__ == '__main__':
    main()







