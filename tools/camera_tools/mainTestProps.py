# \copyright    Sky360.org
#
# \brief        Subscribes to a topic and receives the messages
#               defined by the proto buffer.
#
# ************************************************************************

import cv2
import sys
import numpy as np
import time
import datetime
import argparse

def decode_fourcc(fourcc):
    fourcc = int(fourcc) & 0xFFFFFFFF  # Ensure fourcc is a 32-bit unsigned integer
    fourcc_str = [chr((fourcc >> i) & 0xFF) for i in (0, 8, 16, 24)]
    fourcc_str.append('\0')
    return fourcc_str

def testProps(CameraID, backend, codec, width, height, fps):
    camera.open(CameraID, backend)
    print(f"Camera open: {camera.isOpened()}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))

    print(f"CameraID = {CameraID}")
    print(f"fourcc   = {decode_fourcc(cv2.VideoWriter_fourcc(*codec))} = {decode_fourcc(camera.get(cv2.CAP_PROP_FOURCC))}")
    print(f"Input  Image = [{width}, {height}]; at {fps} fps")
    print(f"Config Image = [{camera.get(cv2.CAP_PROP_FRAME_WIDTH)}, {camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}]; at {camera.get(cv2.CAP_PROP_FPS)} fps")

    # Capture frames for 10 seconds and count how many we get
    start_time = time.time()
    frame_count = 0
    while time.time() - start_time < 10:
        ret, frame = camera.read()
        if ret:
            frame_count += 1
        else:
            return False
    print(f"Real Image  = [{frame.shape[1]}, {frame.shape[0]}]; at {frame_count / 10} fps \n")
    cv2.imshow("image", frame)
    cv2.waitKey(1)
    camera.release()
    return True

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--CameraID", required = False, help = "Specify Camera ID: python mainTestProps.py --CameraID 0")
    args = vars(ap.parse_args())

    # Default configurations
    CameraID = 0;

    print("\n\nDefault usage: python mainTestProps.py --CameraID 0")
    print("\nPress 'Esc' key to stop\n")

    print(" ######################################### ")
    print( decode_fourcc(cv2.VideoWriter_fourcc(*'MJPG')) )
    print( decode_fourcc(cv2.VideoWriter_fourcc(*'DIB ')) )
    print( decode_fourcc(cv2.VideoWriter_fourcc(*'YUY2')) )
    print(" ######################################### ")

    # Windows backends
    # - CAP_DSHOW (DirectShow)
    # - CAP_MSMF (Microsoft Media Foundation)
    # - CAP_VFW (Video For Windows)
    backend = cv2.CAP_DSHOW

    if args["CameraID"]:
        CameraID = int(args["CameraID"])

    # VideoCapture
    camera = cv2.VideoCapture()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # testProps(CameraID, backend, codec, width, height, fps)
    backend = cv2.CAP_DSHOW
    print("\nCAP_DSHOW")
    for CameraID in range(5):
        ret = testProps(CameraID, backend, 'MJPG', 3840, 2448,  30)
        if not ret:
            break
        ret = testProps(CameraID, backend, 'YUY2', 3840, 2448,  30)
        if not ret:
            break

    print("\nCAP_MSMF")
    backend = cv2.CAP_MSMF
    for CameraID in range(5):
        ret = testProps(CameraID, backend, 'MJPG', 3840, 2448,  30)
        if not ret:
            break
        ret = testProps(CameraID, backend, 'YUY2', 3840, 2448,  30)
        if not ret:
            break

    cv2.destroyAllWindows()
