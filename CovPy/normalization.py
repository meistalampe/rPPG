from hdf5_helper_functions import *
import matplotlib.pyplot as plt


def main():
    """
    Loads hdf5 thermal file from directory and extracts image dataset with the corresponding timestamps.
    Returns a set of normalized thermal images.
    :return:
    """
    # Andreas
    #filename = 'ThermalData_18_06_2020_13_19_36.h5'
    # Elena
    filename = 'ThermalData_18_06_2020_13_24_58.h5'
    # Positive
    # filename = 'ThermalData_03_06_2020_11_09_40.h5'
    filepath = 'E:\\GitHub\\CovPySourceFile'
    dataset, timestamps = load_thermal_file(
        _filename=filename,
        _folder=filepath
    )

    # grab first frame for inspection
    plt.imshow(dataset[0])
    plt.show()

    destination_dir = 'E:\\GitHub\\CovPySourceFile\\Normalized'

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # get data set attributes
    n_frames, height, width, total_time_ms = [dataset.attrs[i] for i in list(dataset.attrs)]

    # Andreas
    # x_range = (100, 350)
    # y_range = (100, 420)
    # Elena
    x_range = (200, 420)
    y_range = (0, 300)
    # normalize raw data for further processing steps [0 - 255]
    for n in range(0, n_frames):
        raw_frame = load_sub_frame(dataset, y_range, x_range, n)
        norm_frame = cv2.normalize(
            raw_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # plt.imshow(norm_frame)
        # plt.show()
        cv2.imwrite(destination_dir + '\\NF_{}.png'.format(n), norm_frame)


if __name__ == '__main__':
    main()
