# \copyright    Sky360.org
#
# \brief        Test GMM using a usb camera
#
# ************************************************************************

import cv2
import numpy as np
import argparse

from skimage import morphology
from vibe import ViBe
from timer import Timer


def main(camera_id: np.uint8 = 0, nbr_backgrounds: np.uint8 = 20, min_matches: np.uint8 = 2):
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
    #frame = cv2.resize(frame, (800, 600))

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Foreground (abs diff)", cv2.WINDOW_NORMAL)
    #for i in range(nbr_backgrounds):
    #    cv2.namedWindow(f"{i} - Background (mean)", cv2.WINDOW_NORMAL)

    height, width, depth = frame.shape
    with Timer('gmm init'):
            vibe_bg = ViBe((height, width, depth), nbr_backgrounds, min_matches)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture.")
            return
        # Resize just for demo so that is faster to learn the background
        # frame = cv2.resize(frame, (800, 600))

        # Gaussian Blur - also for demo so that is faster to learn the background
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        frame_u8 = frame.astype(np.uint8)
        frame_u8_normalized = frame_u8

        with Timer('gmm get_difference_mask'):
            foreground = vibe_bg.get_difference_mask(frame_u8)

        # Morphological cleaning (remove small noise and fill holes)
        #masks_bin = foreground.astype(bool)
        #cleaned = morphology.remove_small_objects(masks_bin, min_size=300)
        #cleaned = morphology.remove_small_holes(cleaned, area_threshold=300)
        #cleaned = morphology.binary_closing(cleaned, morphology.disk(5))
        #cleaned = morphology.binary_opening(cleaned, morphology.disk(3))
        #foreground = cleaned.astype(np.uint8) * 255

        with Timer('gmm update'):
            vibe_bg.update(frame_u8_normalized)

        cv2.imshow("Original", frame_u8)
        cv2.imshow("Foreground (abs diff)", foreground)
        #for i in range(nbr_backgrounds):
        #    background = np.clip((vibe_bg.bg_buffer[..., i] + 0.5) * 255, 0, 255).astype(np.uint8)
        #    cv2.imshow(f"{i} - Background (mean)", background)

        # Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python test_gmm.py --camera_id 0")
    ap.add_argument("--nbr_backgrounds", required=False,
                    help="Specify Camera ID: python test_gmm.py --nbr_backgrounds 20")
    ap.add_argument("--min_matches", required=False, type=float,
                    help="Specify Camera ID: python test_gmm.py --min_matches 2")
    args = vars(ap.parse_args())

    # Default configurations
    camera_id = 0
    nbr_backgrounds = 20
    min_matches = 2

    print("\n\nDefault usage: python test_gmm.py --camera_id 0 --nbr_backgrounds 20 --min_matches 2")
    print("\nPress 'Esc' key to stop\n")

    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["nbr_backgrounds"]:
        nbr_backgrounds = int(args["nbr_backgrounds"])
    if args["min_matches"]:
        min_matches = int(args["min_matches"])

    main(camera_id, nbr_backgrounds, min_matches)