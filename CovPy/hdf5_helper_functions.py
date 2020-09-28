"""
Author: Dominik Limbach
Description: handle import and export of hdf5 data
"""

import h5py
import numpy
import cv2

import os
import os.path


def thermal(h5filename, date, destination_dir):
    try:
        thermal_filename = 'Z:\\Thermal\\' + date + '\\' + h5filename
        th5file = h5py.File(thermal_filename, 'r')
        tdata = th5file['FRAMES']
        tstamps_t = th5file['Timestamps_ms']
        n_frames, height, width, total_time_ms = [tdata.attrs[i] for i in list(tdata.attrs)]
        tframe = tdata[200, 0:480, :]
        raw_img = cv2.normalize(tframe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_img = cv2.applyColorMap(raw_img, cv2.COLORMAP_HOT)
        cv2.imwrite(destination_dir + '\\preview_thermal.png', disp_img)
    except FileNotFoundError:
        blank_image = numpy.zeros((480, 640, 3), numpy.uint8)
        cv2.imwrite(destination_dir + '\\blank.png', blank_image)


def load_thermal_file(h5filename, date):
    try:
        filename = 'Z:\\Thermal\\' + date + '\\' + h5filename
        file = h5py.File(filename, 'r')
        data = file['FRAMES']
        timestamps = file['Timestamps_ms']
    except FileNotFoundError:
        data = []
        timestamps = []

    return data, timestamps


def write_frames_to_files(dataset, filename: str = 'ThermalImg_',
                          destination_dir: str = 'E:\\GitHub\\CovPySourceFile\\ThermalImages\\'):

    n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]

    for n in range(0, n_frames):
        frame = dataset[n, 0:height, :]
        raw_img = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_img = cv2.applyColorMap(raw_img, cv2.COLORMAP_HOT)
        cv2.imwrite(destination_dir + filename + '{}.png'.format(n), disp_img)


def images_to_video(image_dir: str = 'E:\\GitHub\\CovPySourceFile\\ThermalImages\\',
                    target_dir: str = 'E:\\GitHub\\CovPySourceFile\\Video\\',
                    image_name_tag: str = 'ThermalImg',
                    video_name: str = 'ThermalVideo',
                    file_type: str = '.png'):

    n_imgs = len([file for file in os.listdir(image_dir) if file.endswith(file_type)])
    img_array = []
    for n in range(0, n_imgs):
        img_name = image_dir + image_name_tag + '_{}'.format(n) + file_type
        img = cv2.imread(img_name)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(target_dir + video_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def load_frame_from_dataset(dataset, frame_height, frame_number):
    return dataset[frame_number, 0:frame_height, :]


