# \copyright    Sky360.org
#
# \brief        Test frame difference using a usb camera
#
# ************************************************************************

import cv2
import numpy as np
import argparse
import time

from skimage import morphology
from frame_difference import FrameDifference

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def main(camera_id: np.uint8 = 0, nbr_history_frame: np.uint8 = 2, is_color: bool = True):
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
    for i in range(nbr_history_frame):
        cv2.namedWindow(f"{i} - Background (mean)", cv2.WINDOW_NORMAL)

    depth=3
    if not is_color:
        depth = 1

    height, width, _ = frame.shape
    with Timer('fd init'):
            fd = FrameDifference((height, width, depth), nbr_history_frame)

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

        with Timer('difference_mask'):
            foreground = fd.process_frame(frame_u8_normalized)

        cv2.imshow("Original", frame_u8)
        cv2.imshow("Foreground (abs diff)", foreground)
        for i in range(nbr_history_frame):
            background = np.clip((fd.frame_history[..., i] + 0.5) * 255, 0, 255).astype(np.uint8)
            cv2.imshow(f"{i} - Background (mean)", background)

        # Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python test_fd.py --camera_id 0")
    ap.add_argument("--nbr_history_frames", required=False,
                    help="Specify Number of history frames: python test_fd.py --nbr_history_frames 2")
    ap.add_argument("--is_color", required=False,
                    help="Select True for color and False for gray : python test_fd.py --is_color True")
    args = vars(ap.parse_args())

    # Default configurations
    camera_id = 0
    nbr_history_frames = 5
    is_color = True

    print("\n\nDefault usage: python test_fd.py --camera_id 0 --nbr_history_frames 2 --is_color True")
    print("\nPress 'Esc' key to stop\n")

    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["nbr_history_frames"]:
        nbr_history_frames = int(args["nbr_history_frames"])
    if args["is_color"]:
        is_color = args["is_color"].lower() == "true"

    main(camera_id, nbr_history_frames, is_color)