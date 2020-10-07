import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv_helper_functions import load_images_from_folder, load_gray_images_from_folder
from skimage.filters import unsharp_mask


def main():
    """
    loads face images, performs edge detection using unsharpened masks and creates ROM.
    :return:
    """
    tag = 'FR_'
    folder = 'E:\\GitHub\\CovPySourceFile\\FaceROI\\'
    faces = load_gray_images_from_folder(
        folder=folder,
        name_tag=tag
    )

    is_drawing = True
    sigma = 0.33
    for n, f in enumerate(faces):
        # # --- smoothen the image ---
        # blur = cv2.GaussianBlur(f, (21, 21), 0)
        #
        # # --- applied the formula assuming amount = 1---
        # u_img = cv2.add(f[:, :], (f[:, :] - blur[:, :]))
        # v = np.median(u_img)
        #
        # # ---- apply automatic Canny edge detection using the computed median----
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        # edged = cv2.Canny(u_img, lower, upper)
        edged = cv2.Canny(f, 100, 100)

        if is_drawing:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
            # Plotting the original image.
            ax[0].imshow(f)
            ax[0].set_title('Thermal Data - Normalized')

            # ax[1].imshow(temp_frames[n])
            # ax[1].set_title('Temp Frame [C]')

            # ax[1].imshow(u_img)
            # ax[1].set_title('Unsharpened Image')

            ax[1].imshow(edged)
            ax[1].set_title('Edges')

            plt.subplots_adjust()
            plt.show()


if __name__ == '__main__':
    main()
