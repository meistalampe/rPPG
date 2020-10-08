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


def apply_gaussian(img, sigma):
    return cv2.GaussianBlur(img, (5, 5), sigmaX=sigma, sigmaY=sigma)


class GetTemperatureFromThermal:

    regions_of_interest = {
        'periorbital': (39, 42),
        'maxillary': (48, 49, 50, 52, 53, 54),
        'nose': (30, 32, 33, 34)
    }

    def __init__(self, h5filename, filepath, savepath):
        self.filename = h5filename
        self.folder = filepath
        self.save_dir = savepath
        self.h5dataset = []
        self.h5timestamps = []
        self.h5attributes = {}
        self.dataset_resampled = []
        self.timestamps_resampled = []
        self.fs = 0
        self.idx_resampled = []
        self.data = []
        self.temp_data = []
        self.x_range_resized = tuple()
        self.y_range_resized = tuple()
        self.landmarks = []
        self.lm_detected = []

    # step 1: load h5 file
    def load_dataset(self):
        with h5py.File(self.folder + '\\' + self.filename, 'r') as f:
            self.h5attributes['width'] = f['FRAMES'].attrs['FrameWidth']
            self.h5attributes['height'] = f['FRAMES'].attrs['FrameHeight']
            self.h5attributes['n_frames'] = f['FRAMES'].attrs['FrameCount']
            self.h5dataset = f['FRAMES'][:]
            timestamps = f['Timestamps_ms'][:]
            # setting the first timestamp to 0:
            self.h5timestamps = np.array(timestamps - timestamps[0]).astype(np.float)

    # step 1.5: down sampling
    def resample(self, fs_new):
        self.timestamps_resampled = np.arange(0, self.h5timestamps[-1], 1000 / fs_new, np.float)
        idx = np.round(np.interp(
            self.timestamps_resampled,
            self.h5timestamps,
            list(range(0, self.h5attributes['n_frames'])))).astype(np.int)

        # resample dataset with new fs
        self.dataset_resampled = self.h5dataset[idx]
        self.idx_resampled = idx

    # step 2: reduce image size
    def set_new_image_size(self):
        # display first frame
        preview = np.zeros((self.h5attributes['height'], self.h5attributes['width'], 3), dtype=np.uint8)
        preview[:, :, 0] = load_frame_from_dataset(
            self.h5dataset,
            self.h5attributes['height'],
            self.h5attributes['width'], 0)

        # Define initial bounding box from roi
        bbox = cv2.selectROI(preview, showCrosshair=True, fromCenter=False)
        self.x_range_resized = (int(bbox[0]), int(bbox[0] + bbox[2]))
        self.y_range_resized = (int(bbox[1]), int(bbox[1] + bbox[3]))
        cv2.destroyAllWindows()

    # step 3: crop images in dataset to new size
    def crop_images(self):
        for n in range(self.h5attributes['n_frames']):
            self.data.append(load_sub_frame(
                frame_number=n,
                dataset=self.h5dataset,
                y_range=self.y_range_resized,
                x_range=self.x_range_resized))

    # step 4: calculate temperature values and store them
    def get_temperature_from_raw(self, dataset):
        temps = []
        for img in dataset:
            temps.append(img * 0.1 - 273.15)

        self.temp_data = temps

    # step 4.5: apply gaussian
    # step 5: normalize thermal data
    def normalize_data(self, dataset):
        norm_data = []
        for img in dataset:
            blur = apply_gaussian(img, sigma=1)
            norm_data.append(cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        self.data = norm_data

    # When Landmarks approach is used
    # step 6: get landmarks for first recognized face (on normalized thermal images)
    def get_face_landmarks(self):
        # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            # flip_input=False,
            device='cpu',
            face_detector='blazeface')
        success = []
        for n, img in enumerate(self.data):
            predictions = fa.get_landmarks_from_image(img)
            print('pred_{}'.format(n))

            if predictions is not None:
                self.landmarks.append(predictions[0])
                success.append(True)
            else:
                self.landmarks.append(np.zeros((68, 2)))
                success.append(False)

        success_rate = success.count(True) / len(success) * 100
        print('Detection successful for {}% of frames.'.format(success_rate))

        self.lm_detected = success
        attributes = {
            'success_rate': success_rate
            # TODO: add confidence scores
        }
        self.write_dataset_to_h5file('lm_detected', self.lm_detected, attributes)

    # step 7:
    def get_temp_of_poi(self, roi_type: str, poi_size: int = 1):
        default = 'periorbital'
        if roi_type in GetTemperatureFromThermal.regions_of_interest.keys():
            poi = GetTemperatureFromThermal.regions_of_interest[roi_type]
        else:
            poi = GetTemperatureFromThermal.regions_of_interest[default]
            print('Unknown region of interest. ROI set to default: ' + default)

        poi_values = []
        for n, det, img, lmv in zip(range(0, len(self.temp_data)), self.lm_detected, self.temp_data, self.data):
            values = []
            if det:
                for p in poi:
                    lm_x = int(self.landmarks[n][p][0])
                    lm_y = int(self.landmarks[n][p][1])

                    area = img[lm_y - poi_size:lm_y + poi_size + 1, lm_x - poi_size:lm_x + poi_size + 1]
                    values.append(np.mean(area))
                    cv2.circle(lmv, (lm_x, lm_y), 2, (0, 255, 0), -1)

                v_avg = np.mean(values)
            else:
                v_avg = 0

            poi_values.append(v_avg)
            cv2.imshow('Landmarks', lmv)

            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

        return poi_values

    def write_dataset_to_h5file(self, setname, dataset, attributes: dict = None):
        with h5py.File(self.save_dir + '\\Results_' + self.filename, 'a') as f:
            if setname in list(f.keys()):
                del f[setname]
            else:
                dset = f.create_dataset(setname, data=dataset)
                if attributes is not None:
                    for key in attributes.keys():
                        dset.attrs[key] = attributes[key]

    def write_landmarks_to_h5file(self, landmarks, attributes: dict = None):
        with h5py.File(self.save_dir + '\\Landmarks_' + self.filename, 'w') as f:
            dset = f.create_dataset('landmarks_thermal', data=landmarks)
            if attributes is not None:
                for key in attributes.keys():
                    dset.attrs[key] = attributes[key]

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

    def run(self, sampling_rate: int = 5, approach: Approaches = Approaches.LANDMARKS):
        # note that the order is important
        # check/initialize destination directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.load_dataset()
        # if precomputed landmarks exist, new image size is matched with the size of the existing landmark image
        if os.path.exists(self.save_dir + '\\Landmarks_' + self.filename):
            with h5py.File(self.save_dir + '\\Landmarks_' + self.filename, 'r') as f:
                self.x_range_resized = f['landmarks_thermal'].attrs['width']
                self.y_range_resized = f['landmarks_thermal'].attrs['height']
        else:
            self.set_new_image_size()
        # only crop after new size has been set
        self.crop_images()
        self.resample(fs_new=sampling_rate)
        self.fs = sampling_rate
        self.get_temperature_from_raw(self.dataset_resampled)
        # write new timestamps to file
        self.write_dataset_to_h5file('timestamps', self.timestamps_resampled)
        self.normalize_data(self.dataset_resampled)
        if not isinstance(approach, Approaches):
            raise TypeError('Note: "approach" must be an instance of Approaches.')

        if approach == Approaches.LANDMARKS:
            # check for precomputed landmarks
            if os.path.exists(self.save_dir + '\\Landmarks_' + self.filename):
                with h5py.File(self.save_dir + '\\Landmarks_' + self.filename, 'r') as f:
                    self.landmarks = f['landmarks_thermal']
            else:
                self.get_face_landmarks()
                lm_atts = {
                    'height': self.y_range_resized,
                    'width': self.x_range_resized,
                    'idx': self.idx_resampled,
                    'timestamps': self.timestamps_resampled
                }
                self.write_landmarks_to_h5file(self.landmarks, lm_atts)

            roi_types = list(GetTemperatureFromThermal.regions_of_interest.keys())

            for rt in roi_types:
                temps = self.get_temp_of_poi(roi_type=rt)
                # plt.plot(temps)
                # plt.title(rt)
                # plt.ylabel('Temperature [Celsius]')
                # plt.xlabel('Sample')
                # plt.show()
                t_attributes = {
                    'fs': self.fs,
                    'n': len(temps),
                    'roi_type': rt,
                    'unit': 'Celsius'
                }
                self.write_dataset_to_h5file(setname=rt, dataset=temps, attributes=t_attributes)

            # write lm verification dset
            lm_attributes = {
                'height': self.y_range_resized,
                'width': self.x_range_resized,
                'idx': self.idx_resampled,
                'timestamps': self.timestamps_resampled
            }
            self.write_dataset_to_h5file('lm_verification', self.data, attributes=lm_attributes)

        else:
            self.get_otsu_masks()


# A
file_name = 'ThermalData_18_06_2020_13_19_36.h5'
# E
# file_name = 'ThermalData_18_06_2020_13_24_58.h5'
# P
# file_name = 'ThermalData_03_06_2020_11_09_40.h5'
folder = 'E:\\GitHub\\CovPySourceFile'

destination_dir = 'E:\\GitHub\\CovPySourceFile\\Results'
# destination_dir = 'E:\\GitHub\\rPPG\\Results'
t = GetTemperatureFromThermal(file_name, folder, destination_dir)
t.run(approach=Approaches.LANDMARKS)

