# \copyright    Sky360.org
#
# \brief        Test for Object Detection.
#
# ************************************************************************

import cv2
import numpy as np
import argparse
from object_detector import ObjectDetector
from timer import Timer

def main(camera_id: np.uint8 = 0, nbr_history_frame: np.uint8 = 2):
    cap = cv2.VideoCapture(camera_id)

    # Set auto exposure to off (mode 1) or manual mode (mode 0.75)
    # Note: This might not work with all cameras
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)

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
    # frame = cv2.resize(frame, (800, 600))

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mag_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ang_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("rgb_diff_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blobs_dog_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("gmm_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("vibe_mask", cv2.WINDOW_NORMAL)

    # Object detector
    od = ObjectDetector(frame.shape)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture.")
            return

        # Resize just for demo so that is faster to learn the background
        # frame = cv2.resize(frame, (800, 600))

        # Run the detector
        with Timer('Object detector'):
            od.process_frame(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("mag_mask", od.mag_mask.astype(np.uint8))
        cv2.imshow("ang_mask", od.ang_mask.astype(np.uint8))
        cv2.imshow("rgb_diff_mask", od.rgb_diff_mask.astype(np.uint8))
        cv2.imshow("blobs_dog_mask", od.blobs_dog_mask.astype(np.uint8))
        cv2.imshow("gmm_mask", od.gmm_mask.astype(np.uint8))
        cv2.imshow("vibe_mask", od.vibe_mask.astype(np.uint8))

        # Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python object_detector.py --camera_id 0")
    ap.add_argument("--nbr_history_frames", required=False,
                    help="Specify Number of history frames: python object_detector.py --nbr_history_frames 2")
    args = vars(ap.parse_args())

    # Default configurations
    camera_id = 0
    nbr_history_frames = 3

    print("\n\nDefault usage: python object_detector.py --camera_id 0 --nbr_history_frames 2")
    print("\nPress 'Esc' key to stop\n")

    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["nbr_history_frames"]:
        nbr_history_frames = int(args["nbr_history_frames"])

    main(camera_id, nbr_history_frames)
