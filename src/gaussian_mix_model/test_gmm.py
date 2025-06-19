# \copyright    Sky360.org
#
# \brief        Test GMM using a usb camera
#
# ************************************************************************

import cv2
import numpy as np
import argparse
import time

from skimage import morphology
from gaussian_mix_models import GaussianMixModels

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def main(camera_id: np.uint8 = 0, max_nbr_gaussians: np.uint8 = 5, learning_factor: np.float32 = 0.5, is_color: bool = True):
    cap = cv2.VideoCapture(camera_id)

    # Set auto exposure to off (mode 1) or manual mode (mode 0.75)
    # Note: This might not work with all cameras
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5448)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print(f"Check configuration: ")
    print(f"camera_id = {camera_id}")
    print(
        f"Config Image = [{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}]; at {cap.get(cv2.CAP_PROP_FPS)} fps")

    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera.")
        return

    # Resize just for demo so that is faster to learn the background
    frame = cv2.resize(frame, (800, 600))
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Foreground (abs diff)", cv2.WINDOW_NORMAL)
    for i in range(max_nbr_gaussians):
        cv2.namedWindow(f"{i} - Background (mean)", cv2.WINDOW_NORMAL)

    depth=3
    if not is_color:
        depth = 1

    height, width, _ = frame.shape
    with Timer('gmm init'):
            gmm = GaussianMixModels((height, width, depth), max_nbr_gaussians, learning_factor)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture.")
            return
        # Resize just for demo so that is faster to learn the background
        frame = cv2.resize(frame, (800, 600))
        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur - also for demo so that is faster to learn the background
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        frame_u8 = frame.astype(np.uint8)
        frame_u8_normalized = frame_u8
        if is_color:
            frame_u8_normalized = frame_u8[:, :, :3]  # Make sure it's 3 channels

        with Timer('gmm get_difference_mask'):
            foreground = gmm.get_difference_mask(frame_u8)

        # Morphological cleaning (remove small noise and fill holes)
        #masks_bin = foreground.astype(bool)
        #cleaned = morphology.remove_small_objects(masks_bin, min_size=300)
        #cleaned = morphology.remove_small_holes(cleaned, area_threshold=300)
        #cleaned = morphology.binary_closing(cleaned, morphology.disk(5))
        #cleaned = morphology.binary_opening(cleaned, morphology.disk(3))
        #foreground = cleaned.astype(np.uint8) * 255

        with Timer('gmm update'):
            gmm.update(frame_u8_normalized)

        cv2.imshow("Original", frame_u8)
        cv2.imshow("Foreground (abs diff)", foreground)
        for i in range(max_nbr_gaussians):
            background = np.clip((gmm.mean[..., i] + 0.5) * 255, 0, 255).astype(np.uint8)
            cv2.imshow(f"{i} - Background (mean)", background)

        # Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python test_gmm.py --camera_id 0")
    ap.add_argument("--max_nbr_gaussians", required=False,
                    help="Specify Camera ID: python test_gmm.py --max_nbr_gaussians 5")
    ap.add_argument("--learning_factor", required=False, type=float,
                    help="Specify Camera ID: python test_gmm.py --learning_factor 0.5")
    ap.add_argument("--is_color", required=False,
                    help="Select True for color and False for gray : python test_gmm.py --is_color True")
    args = vars(ap.parse_args())

    # Default configurations
    camera_id = 0
    is_color = True
    max_nbr_gaussians = 13
    learning_factor = 0.3

    print("\n\nDefault usage: python test_gmm.py --camera_id 0 --max_nbr_gaussians 5 --learning_factor 0.5 --is_color True")
    print("\nPress 'Esc' key to stop\n")

    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["is_color"]:
        is_color = args["is_color"].lower() == "true"
    if args["max_nbr_gaussians"]:
        max_nbr_gaussians = int(args["max_nbr_gaussians"])
    if args["learning_factor"]:
        learning_factor = args["learning_factor"]

    main(camera_id, max_nbr_gaussians, learning_factor, is_color)