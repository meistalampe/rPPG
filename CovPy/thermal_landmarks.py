import h5py
import cv2
import os
from os.path import join as fullfile
import numpy as np
import face_alignment


def apply_gaussian(img, sigma):
    return cv2.GaussianBlur(img, (5, 5), sigmaX=sigma, sigmaY=sigma)


class LandmarkGenerator:
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
        self.landmarks = []

    def load_dataset(self):
        with h5py.File(fullfile(self.folder, self.filename), 'r') as f:
            self.h5attributes['width'] = f['FRAMES'].attrs['FrameWidth']
            self.h5attributes['height'] = f['FRAMES'].attrs['FrameHeight']
            self.h5attributes['n_frames'] = f['FRAMES'].attrs['FrameCount']
            self.h5dataset = f['FRAMES'][:]
            timestamps = f['Timestamps_ms'][:]
            # setting the first timestamp to 0:
            self.h5timestamps = np.array(timestamps - timestamps[0]).astype(np.float)

    def resample(self, fs_new):
        self.timestamps_resampled = np.arange(0, self.h5timestamps[-1], 1000 / fs_new, np.float)
        idx = np.round(np.interp(
            self.timestamps_resampled,
            self.h5timestamps,
            list(range(0, self.h5attributes['n_frames'])))).astype(np.int)

        # resample dataset with new fs
        self.dataset_resampled = self.h5dataset[idx]
        self.idx_resampled = idx

    def initialize_landmark_dataset(self):
        lm_mask = np.full((68, 2), -1)
        max_faces = 10
        signal_length = len(self.timestamps_resampled)
        for n in range(0, max_faces):
            line = [lm_mask] * signal_length
            self.landmarks.append(line)

    # apply gaussian
    # normalize thermal data
    def normalize_data(self, dataset):
        norm_data = []
        for img in dataset:
            blur = apply_gaussian(img, sigma=1)
            norm_data.append(cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        self.data = norm_data

    def get_face_landmarks(self):
        # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            # flip_input=False,
            device='cpu',
            face_detector='blazeface')

        for n, img in enumerate(self.data):
            predictions = fa.get_landmarks_from_image(img)
            if predictions is not None:
                for p, pred in enumerate(predictions):
                    self.landmarks[p][n] = pred

    def write_landmarks_to_h5file(self, landmarks, attributes: dict = None):
        with h5py.File(fullfile(self.save_dir, 'Landmarks_' + self.filename), 'w') as f:
            dset = f.create_dataset('landmarks_thermal', data=landmarks)
            if attributes is not None:
                for key in attributes.keys():
                    dset.attrs[key] = attributes[key]

    def run(self, sampling_rate: int = 5):
        # note that the order is important
        # check/initialize destination directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.load_dataset()
        self.resample(fs_new=sampling_rate)
        self.initialize_landmark_dataset()
        self.normalize_data(self.dataset_resampled)
        self.get_face_landmarks()
        lm_atts = {
            'height': self.h5attributes['height'],
            'width': self.h5attributes['width'],
            'idx': self.idx_resampled,
            'timestamps': self.timestamps_resampled,
        }
        self.write_landmarks_to_h5file(self.landmarks, lm_atts)


class LandmarkVerification:
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
        self.lm_verification = []

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

    def draw_face_from_landmarks(self, face_number: int = 0, verbose: bool = False):
        face_landmarks = self.landmarks[face_number]
        verification_imgs = []
        for i, lm in enumerate(face_landmarks):
            img = cv2.normalize(
                self.dset[self.lm_attrs['idx'][i]], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            for (x, y) in lm:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

            verification_imgs.append(img)

            if verbose:
                cv2.imshow('LM Verification', img)

                k = cv2.waitKey(1)
                if k == 27:
                    break

        cv2.destroyAllWindows()

        self.lm_verification = verification_imgs

    def write_dataset_to_h5file(self, setname, dataset, attributes: dict = None):
        subject_id = self.lm_file.partition("ThermalData_")[2].partition(".h5")[0]
        name = 'Landmark_Verification_' + subject_id + '_F{}_'.format(self.lm_attrs['face_number'])
        with h5py.File(fullfile(self.save_dir, name), 'w') as f:
            dset = f.create_dataset(setname, data=dataset)
            if attributes is not None:
                for key in attributes.keys():
                    dset.attrs[key] = attributes[key]

    def run(self, face_number: int = 0):
        self.load_dataset()
        self.load_landmarks()
        is_matching = self.name_inspector()
        is_same_size = self.dim_inspector()
        if is_matching and is_same_size:
            self.lm_attrs['face_number'] = face_number
            self.draw_face_from_landmarks(face_number=face_number, verbose=True)
            self.write_dataset_to_h5file('lm_verification', self.lm_verification, self.lm_attrs)
        else:
            # as both inspectors should raise an exception if there is a mismatch the program should never get here
            return -1


# Andreas
file_name = 'ThermalData_18_06_2020_13_19_36.h5'
# Elena
# file_name = 'ThermalData_18_06_2020_13_24_58.h5'
folder = 'E:\\GitHub\\CovPySourceFile'
destination_dir = 'E:\\GitHub\\CovPySourceFile\\Results'

# lg = LandmarkGenerator(file_name, folder, destination_dir)
# lg.run()

lm_name = 'Landmarks_ThermalData_18_06_2020_13_19_36.h5'
lv = LandmarkVerification(
    data_filename=file_name,
    data_path=folder,
    lm_filename=lm_name,
    lm_path=destination_dir,
    savepath=destination_dir
)
lv.run(face_number=0)


