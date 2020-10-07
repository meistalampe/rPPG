import h5py
import cv2
import itertools
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
from enum import Enum
from hdf5_helper_functions import *
from cv_helper_functions import *


class Approaches(Enum):
    LANDMARKS = 1,
    LAYERING = 2


class GetTemperatureFromThermal:

    regions_of_interest = {
        'periorbital': (39, 42),
        'maxillary': (48, 49, 50, 52, 53, 54),
        'nosetip': (30, 32, 33, 34)
    }

    class ImageList:

        def __init__(self, images, height, width):
            self.images = images
            self.height = height
            self.width = width

    def __init__(self, h5filename, filepath):
        self.filename = h5filename
        self.folder = filepath
        self.h5dataset = []
        self.h5timestamps = []
        self.data = []
        self.temp_data = []
        self.x = tuple()
        self.y = tuple()
        self.landmarks = []
        self.lm_detected = []

    # step 1: load h5 file
    def load_dataset(self):
        self.h5dataset, self.h5timestamps = load_thermal_file(
            _filename=self.filename,
            _folder=self.folder
        )

    # step 2: reduce image size
    def set_new_image_size(self):
        # get data set attributes
        n_frames, height, width, total_time_ms = [self.h5dataset.attrs[i] for i in list(self.h5dataset.attrs)]
        # display first frame
        preview = np.zeros((height, width, 3), dtype=np.uint8)
        preview[:, :, 0] = load_frame_from_dataset(self.h5dataset, height, width, 0)

        # Define initial bounding box from roi
        bbox = cv2.selectROI(preview, showCrosshair=True, fromCenter=False)
        self.x = (int(bbox[0]), int(bbox[0] + bbox[2]))
        self.y = (int(bbox[1]), int(bbox[1] + bbox[3]))
        cv2.destroyAllWindows()

    # step 3: crop images in dataset to new size
    def crop_images(self):
        # get data set attributes
        n_frames, height, width, total_time_ms = [self.h5dataset.attrs[i] for i in list(self.h5dataset.attrs)]
        for n in range(n_frames):
            self.data.append(load_sub_frame(frame_number=n, dataset=self.h5dataset, y_range=self.y, x_range=self.x))

    # step 4: calculate temperature values and store them
    def get_temperature_from_raw(self):
        temps = []
        for img in self.data:
            temps.append(img * 0.1 - 273.15)

        self.temp_data = temps

    # step 5: normalize thermal data
    def normalize_data(self):
        norm_data = []
        for img in self.data:
            norm_data.append(cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        self.data = norm_data

    # When Landmarks approach is used
    # step 6: get landmarks for first recognized face (on normalized thermal images)
    def get_face_landmarks(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        success = []
        for n, img in enumerate(self.data):
            predictions = fa.get_landmarks_from_image(img)

            if predictions is not None:
                # extract landmarks
                self.landmarks.append(predictions[0])
                success.append(True)
            else:
                self.landmarks.append(np.empty((1, 1), dtype=np.uint8))
                success.append(False)

        success_rate = success.count(True) / len(success) * 100
        print('Detection successful for {}% of frames.'.format(success_rate))

        self.lm_detected = success

    # step 7:
    def get_temp_of_poi(self, roi_type: str):
        default = 'periorbital'
        if roi_type in GetTemperatureFromThermal.regions_of_interest.keys():
            poi = GetTemperatureFromThermal.regions_of_interest[roi_type]
        else:
            poi = GetTemperatureFromThermal.regions_of_interest[default]
            print('Unknown region of interest. ROI set to default: ' + default)

        poi_values = []
        for n, det, img in zip(range(0, len(self.temp_data)), self.lm_detected, self.temp_data):
            values = []
            if det:
                for p in poi:
                    lm_x = int(self.landmarks[n][p][0])
                    lm_y = int(self.landmarks[n][p][1])
                    values.append(img[lm_y, lm_x])
                    # cv2.circle(img, (lm_x, lm_y), 2, (0, 255, 0), -1)

                v_avg = np.mean(values)
            else:
                v_avg = 0

            poi_values.append(v_avg)
            # cv2.imshow('Landmarks', img)
            #
            # k = cv2.waitKey(1)
            # if k == 27:
            #     break

        cv2.destroyAllWindows()

        return poi_values

    def write_dataset_to_h5file(self, setname, dataset):
        with h5py.File('Results_' + self.filename, 'a') as f:
            f.create_dataset(setname.upper(), data=dataset)

    # When Layering Approach is used
    # step 6: get otsu masks
    def get_otsu_masks(self):
        print('This path has not been implemented yet.')

    def show_data(self):
        for n in range(0, len(self.data)):
            cv2.imshow('Hallo', self.data[n])

            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

    def run(self, approach: Approaches = Approaches.LANDMARKS):
        # note that the order is important
        self.load_dataset()
        self.set_new_image_size()
        # only crop after new size has been set
        self.crop_images()
        self.get_temperature_from_raw()
        self.normalize_data()
        if not isinstance(approach, Approaches):
            raise TypeError('approach must be an instance of Approaches.')

        if approach == Approaches.LANDMARKS:
            self.get_face_landmarks()
            roi_types = ['maxillary', 'nosetip', 'periorbital']
            for rt in roi_types:
                temps = self.get_temp_of_poi(roi_type=rt)
                plt.plot(temps)
                plt.title(rt)
                plt.ylabel('Temperature [Celsius]')
                plt.xlabel('Sample')
                plt.show()
                self.write_dataset_to_h5file(setname=rt, dataset=temps)

            self.write_dataset_to_h5file('timestamps', self.h5timestamps)

        else:
            self.get_otsu_masks()


# A
file_name = 'ThermalData_18_06_2020_13_19_36.h5'
folder = 'E:\\GitHub\\CovPySourceFile'
t = GetTemperatureFromThermal(file_name, folder)
t.run(approach=Approaches.LANDMARKS)

