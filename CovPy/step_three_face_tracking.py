import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_helper_functions import load_images_from_folder, get_values_from_roi
from algorithm_functions import *
from skimage.filters import unsharp_mask
from PIL import Image


def main():
    """
    LoadsMulti-Level Otsu masks and tracks the face.
    Returns a set of ROIs containing only the face.
    :return:
    """
    tag = 'MLO_'
    filepath = 'E:\\GitHub\\CovPySourceFile\\MultiLevelOtsu\\'
    otsu_masks = load_images_from_folder(
        folder=filepath,
        name_tag=tag,
    )

    # region MIL Tracking
    # use binary otsu mask to detect the face
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    # Set up tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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

    initial_frame = otsu_masks[0]
    # Define initial bounding box from roi
    bbox = cv2.selectROI(initial_frame, showCrosshair=True, fromCenter=False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(initial_frame, bbox)

    # roi points
    points = []
    failed_idx = []
    for n, mask in enumerate(otsu_masks):
        # Update tracker
        ok, bbox = tracker.update(mask)
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = [int(bbox[0]), int(bbox[1])]
            p2 = [int(bbox[0] + bbox[2]), int(bbox[1])]
            p3 = [int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
            p4 = [int(bbox[0]), int(bbox[1] + bbox[3])]
            points.append([p1, p2, p3, p4])

            cv2.rectangle(mask, tuple(p1), tuple(p3), (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(
                mask, "Tracking failure detected",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            failed_idx.append(n)

        # Display result
        cv2.imshow("Tracking", mask)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    # endregion

    # get rois
    rois = []
    for n, rp in enumerate(points):
        img = Image.open(r"E:\GitHub\CovPySourceFile\Normalized\NF_{}.png".format(n))
        left = rp[0][0]
        top = rp[0][1]
        right = rp[2][0]
        bottom = rp[2][1]
        cropped = img.crop((left, top, right, bottom))
        rois.append(cropped)

        # plt.imshow(cropped)
        # plt.show()

    destination_dir = 'E:\\GitHub\\CovPySourceFile\\FaceROI\\'

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for n, r in enumerate(rois):
        r.save(destination_dir + 'FR_{}.png'.format(n))


if __name__ == '__main__':
    main()
