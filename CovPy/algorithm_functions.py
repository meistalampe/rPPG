import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from enum import Enum
from cv_helper_functions import *
from hdf5_helper_functions import *

from skimage import data
from skimage.filters import threshold_multiotsu


class ThresholdingMethods(Enum):
    CORNERS = 1
    MINIMUM = 2


class KernelMethods(Enum):
    SOBEL = 1
    LAPLACIAN = 2
    CANNY = 3


class OtsuMethods(Enum):
    IMAGES = 1
    BINARY = 2


def multi_level_otsu(
        images,
        n_regions: int = 4,
        target_region: int = 3,
        method: OtsuMethods = OtsuMethods.IMAGES,
        _destination_dir: str = 'E:\\GitHub\\CovPySourceFile\\MultiLevelOtsu',
        draw: bool = False,
        write: bool = False):
    """
    Function applies multi-level Otsu algorithm to seperate the image background from the foreground.

    :param images:
        input the normalized thermal data.
    :param n_regions:
        number of layers the algorithm tries to create.
    :param target_region:
        number of the closest layer. This is the foreground and should contain the desired face regions.
    :param method:
        determines if the mask is filled binary or with actual values
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

    if not isinstance(method, OtsuMethods):
        raise TypeError('Method must be an instance of OtsuMethods.')

    if not os.path.exists(_destination_dir):
        os.makedirs(_destination_dir)

    otsu_images = []
    # apply multi-level Otsu threshold for the input value n_regions, generating just as many classes
    for th_n, f in enumerate(images):
        # start = time.time()
        thresholds = threshold_multiotsu(image=f, classes=n_regions)
        # use the threshold values to generate the classes
        regions = np.digitize(f, bins=thresholds)

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
        if method == OtsuMethods.IMAGES:
            otsu_mask = np.where(regions == target_region, f, 0)
        else:
            otsu_mask = np.where(regions == target_region, 1, 0)

        otsu_images.append(otsu_mask)

        # end = time.time()
        # print('MLO: Frame processed in {}'.format(end - start))

        if draw:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 3.5))

            # plotting the original image.
            ax[0][0].imshow(f, cmap='gray')
            ax[0][0].set_title('Original')

            # plotting the target region.
            ax[0][1].imshow(otsu_mask)
            ax[0][1].set_title('Target Region')

            # plotting the histogram and the two thresholds obtained from multi-Otsu.
            ax[1][0].hist(f.ravel(), bins=255)
            ax[1][0].set_title('Histogram')
            for thresh in thresholds:
                ax[1][0].axvline(thresh, color='r')

            # Plotting the Multi Otsu result.
            ax[1][1].imshow(regions, cmap='Accent')
            ax[1][1].set_title('Multi-Otsu result')

            plt.subplots_adjust()
            plt.show()

        if write:
            cv2.imwrite(_destination_dir + '\\MLO_{}.png'.format(th_n), otsu_mask)

    if draw:
        plt.close('all')

    return otsu_images


def optimal_quantization(dataset, method: ThresholdingMethods = ThresholdingMethods.CORNERS, write: bool = True,
                         save_path: str = 'E:\\GitHub\\CovPySourceFile\\OptimizedImages\\'):
    """
    Optimal Quantization – convert from the absolute temperature distributions to
    the color-mapped images by analyzing the temperature histogram of every frame
    :return
    img_array: list
    """
    if not isinstance(method, ThresholdingMethods):
        raise TypeError('Method must be an instance of ThresholdingMethods.')

    n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]
    img_array = []

    for n in range(0, n_frames):
        # load frame
        frame = load_frame_from_dataset(dataset=dataset, frame_height=height, frame_number=n)
        frame = frame * 0.1 - 273.15
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Optimal Quantization for high thermal dynamic range scenes
        f_mean = np.mean(frame)
        f_std = np.std(frame)

        t_max = f_mean + 1.96 * (f_std / math.sqrt(height * width))
        # Ridler and Calvard’s concept of optimal threshold selection

        if method == ThresholdingMethods.CORNERS:
            # try original attempt and set t_opt as avg of corner arrays
            roi_size_factor = 0.1
            lu_roi = frame[1:int(height * roi_size_factor), 1:int(width * roi_size_factor)]
            ll_roi = frame[-int(height * roi_size_factor):-1, 1:int(width * roi_size_factor)]
            ru_roi = frame[1:int(height * roi_size_factor), -int(width * roi_size_factor):-1]
            rl_roi = frame[-int(height * roi_size_factor):-1, -int(width * roi_size_factor):-1]
            roi_mean = np.mean(np.concatenate((lu_roi, ll_roi, ru_roi, rl_roi)))

            t_opt = roi_mean
        else:
            # initialize the optimal temperature threshold as t_min
            t_min = f_mean - 1.96 * (f_std / math.sqrt(height * width))
            t_opt = t_min

        # print(t_opt)
        # iterate until t_threshold_optimal(i) - t_threshold_optimal(i-1) ~ 0
        while True:
            t_opt_old = t_opt
            mu_lower = np.mean(frame[frame < t_opt])
            mu_higher = np.mean(frame[frame > t_opt])
            t_opt = 0.5 * (mu_lower + mu_higher)
            # print("T_opt(p): {}    T_opt(p-1): {}".format(t_opt, t_opt_old))

            if math.isclose(t_opt, t_opt_old, rel_tol=1e-5):
                break

        t_range = (t_opt, t_max)
        # print(t_range)

        final_img = np.where(frame >= t_range[0], frame, 0)
        img_array.append(final_img)
        # img_normalized = cv2.applyColorMap(img_normalized, cv2.COLORMAP_HOT)
        if write:
            cv2.imwrite(save_path + 'OptImg_{}.png'.format(n), final_img)

    return img_array


def gradiant_mapping(images, kernel_type: KernelMethods = KernelMethods.LAPLACIAN, kernel_size: int = 3,
                     write: bool = False, save_path: str = 'E:\\GitHub\\CovPySourceFile\\GradiantMagnitudeMaps\\'):

    if not isinstance(kernel_type, KernelMethods):
        raise TypeError("Unknown KernelType.")

    magnitude_map = []

    for n, frame in enumerate(images):
        # Remove noise by blurring with a Gaussian filter
        if kernel_type == KernelMethods.CANNY:
            src = frame
        else:
            src = cv2.GaussianBlur(frame, (9, 9), 0)

        if kernel_type == KernelMethods.SOBEL:
            sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=kernel_size)
            mag_map = sobelx + sobely

        elif kernel_type == KernelMethods.CANNY:
            mag_map = cv2.Canny(src, 90, 90, L2gradient=True)
        else:
            mag_map = cv2.Laplacian(src, cv2.CV_64F, ksize=kernel_size)

        mag_map = cv2.convertScaleAbs(mag_map)
        magnitude_map.append(mag_map)

        if write:
            cv2.imwrite(save_path + 'MagMap_{}.png'.format(n), mag_map)

    return magnitude_map


def thermal_gradient_flow():
    """
    Thermal Gradient Flow – nostril-region tracking method using the thermal gradient magnitude and
    points tracking methods
    """
    pass


def thermal_voxel_respiration_estimation():
    """
    Thermal Voxel-based Respiratory Rate Estimation – extracting
    the respiratory signals by integrating the unit thermal voxels inside the nostril
    """
    pass
