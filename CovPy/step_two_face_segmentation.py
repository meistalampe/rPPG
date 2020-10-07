import time
import cv2
from cv_helper_functions import load_images_from_folder, load_gray_images_from_folder
from algorithm_functions import *


def main():
    """
    Use Multi-Level Otsu method to separate face pixels from background.
    Returns a set of binary masks.
    :return:
    """
    tag = 'NF_'
    folder = 'E:\\GitHub\\CovPySourceFile\\Normalized\\'
    thermal_images = load_images_from_folder(
        folder=folder,
        name_tag=tag,
    )

    destination_dir = 'E:\\GitHub\\CovPySourceFile\\MultiLevelOtsu'
    is_writing = False
    is_drawing = True

    start = time.time()

    otsu_masks = multi_level_otsu(
        images=thermal_images,
        n_regions=4,
        target_region=1,
        method=OtsuMethods.IMAGES,
        _destination_dir=destination_dir,
        write=is_writing,
        draw=is_drawing)

    end = time.time()
    print('MLO: Frame processed in {}'.format(end - start))

    
if __name__ == '__main__':
    main()
