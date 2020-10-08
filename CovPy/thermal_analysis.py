import h5py
import cv2
import os
from os.path import join as fullfile
import numpy as np
import face_alignment
from thermal_landmarks import LandmarkGenerator, LandmarkVerification, apply_gaussian


class ThermalAnalysis:

    regions_of_interest = {
        'periorbital': (39, 42),
        'maxillary': (48, 49, 50, 52, 53, 54),
        'nose': (30, 32, 33, 34)
    }

    def __init__(self, data_filename: str, lm_filename: str, data_path: str, lm_path: str, savepath: str):
        self.data_file = data_filename
        self.lm_file = lm_filename
        self.data_folder = data_path
        self.lm_folder = lm_path
        self.save_dir = savepath
        self.dset = []
        self.dset_attrs = {}
        self.timestamps = []
        self.landmarks = []
        self.lm_attrs = {}
        self.temp_data = []

    def name_inspector(self):
        """
        Check if data file and landmark file are from the same subject.
        """
        check_lm = self.lm_file.partition("ThermalData_")[2].partition(".h5")[0]
        check_data = self.data_file.partition("ThermalData_")[2].partition(".h5")[0]

        if check_data != check_lm:
            raise Exception('File names do not match! Image data and Landmarks belong to different subjects.')
        else:
            return True

    def dim_inspector(self):
        """
        Check if data and landmarks are compatible in terms of width and height.
        """
        if self.dset_attrs['width'] != self.lm_attrs['width']:
            raise Exception('Dimension Error.The width attribute of the dataset is not compatible with the landmarks.')
        elif self.dset_attrs['height'] != self.lm_attrs['height']:
            raise Exception('Dimension Error.The height attribute of the dataset is not compatible with the landmarks.')
        else:
            return True

    def load_dataset(self):
        with h5py.File(fullfile(self.data_folder, self.data_file), 'r') as f:
            self.dset_attrs['width'] = f['FRAMES'].attrs['FrameWidth']
            self.dset_attrs['height'] = f['FRAMES'].attrs['FrameHeight']
            self.dset_attrs['n_frames'] = f['FRAMES'].attrs['FrameCount']
            self.dset = f['FRAMES'][:]
            timestamps = f['Timestamps_ms'][:]
            # setting the first timestamp to 0:
            self.timestamps = np.array(timestamps - timestamps[0]).astype(np.float)

    def load_landmarks(self):
        with h5py.File(fullfile(self.lm_folder, self.lm_file), 'r') as f:
            self.lm_attrs['width'] = f['landmarks_thermal'].attrs['width']
            self.lm_attrs['height'] = f['landmarks_thermal'].attrs['height']
            self.lm_attrs['idx'] = f['landmarks_thermal'].attrs['idx']
            self.lm_attrs['timestamps'] = f['landmarks_thermal'].attrs['timestamps']
            self.landmarks = f['landmarks_thermal'][:]

    def get_temperature_from_raw(self, dataset):
        temps = []
        for img in dataset:
            temps.append(img * 0.1 - 273.15)

        self.temp_data = temps

    def get_temp_of_poi(self, face_number: int = 0, roi_type: str = 'periorbital', poi_size: int = 1):
        if roi_type in ThermalAnalysis.regions_of_interest.keys():
            poi = ThermalAnalysis.regions_of_interest[roi_type]
        else:
            raise ValueError('Unknown ROI type.')

        face_landmarks = self.landmarks[face_number]
        poi_values = []
        timestamps = []
        for i, lm in enumerate(face_landmarks):
            img = self.temp_data[self.lm_attrs['idx'][i]]
            timestamps.append(self.timestamps[self.lm_attrs['idx'][i]])
            values = []
            # check if landmark is 'empty' i.e. it contains the default value of -1 in any coordinate 
            if -1 in lm:
                v_avg = 0
            else:
                for n, (x, y) in enumerate(lm):
                    if n in poi:
                        area = img[int(y) - poi_size:int(y) + poi_size + 1, int(x) - poi_size:int(x) + poi_size + 1]
                        values.append(np.mean(area))

                v_avg = np.mean(values)
            poi_values.append(v_avg)

        return poi_values, timestamps

    def write_dataset_to_h5file(self, setname, dataset, attributes: dict = None):
        subject_id = self.data_file.partition("ThermalData_")[2].partition(".h5")[0]
        name = 'Results_' + subject_id + '_F{}_'.format(self.lm_attrs['face_number'])
        with h5py.File(fullfile(self.save_dir, name), 'a') as f:
            if setname in list(f.keys()):
                del f[setname]
            else:
                dset = f.create_dataset(setname, data=dataset)
                if attributes is not None:
                    for key in attributes.keys():
                        dset.attrs[key] = attributes[key]

    def run(self, face_number: int = 0):
        self.load_dataset()
        self.load_landmarks()
        self.get_temperature_from_raw(dataset=self.dset)
        roi_types = list(ThermalAnalysis.regions_of_interest.keys())

        for rt in roi_types:
            temperatures, timestamps = self.get_temp_of_poi(face_number=face_number, roi_type=rt, poi_size=1)
            t_attributes = {
                'timestamps': timestamps,
                'n_frames': len(temperatures),
                'roi_type': rt,
                'unit': 'Celsius'
            }
            self.write_dataset_to_h5file(setname=rt, dataset=temperatures, attributes=t_attributes)
