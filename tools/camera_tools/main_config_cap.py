# \copyright    Sky360.org
#
# \brief        Subscribes to a topic and receives the messages
#               defined by the proto buffer.
#
# ************************************************************************

import sys
import time
import cv2
import argparse

def main(CameraID):

    # Create a windows
    cv2.namedWindow("Sliders")
    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)

    # Open the webcam
    # Windows backends
    # - CAP_DSHOW (DirectShow)
    # - CAP_MSMF (Microsoft Media Foundation)
    # - CAP_VFW (Video For Windows)
    camera = cv2.VideoCapture(CameraID, cv2.CAP_MSMF)
    print(f"Camera open: {camera.isOpened()}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print(f"CAP_PROP_BRIGHTNESS                  = {camera.get(cv2.CAP_PROP_BRIGHTNESS           )}")
    print(f"CAP_PROP_CONTRAST                    = {camera.get(cv2.CAP_PROP_CONTRAST             )}")
    print(f"CAP_PROP_SATURATION                  = {camera.get(cv2.CAP_PROP_SATURATION           )}")
    print(f"CAP_PROP_HUE                         = {camera.get(cv2.CAP_PROP_HUE                  )}")
    print(f"CAP_PROP_GAIN                        = {camera.get(cv2.CAP_PROP_GAIN                 )}")
    print(f"CAP_PROP_EXPOSURE                    = {camera.get(cv2.CAP_PROP_EXPOSURE             )}")
    print(f"CAP_PROP_AUTO_EXPOSURE               = {camera.get(cv2.CAP_PROP_AUTO_EXPOSURE        )}")
    print(f"CAP_PROP_FRAME_WIDTH                 = {camera.get(cv2.CAP_PROP_FRAME_WIDTH          )}")
    print(f"CAP_PROP_FRAME_HEIGHT                = {camera.get(cv2.CAP_PROP_FRAME_HEIGHT         )}")
    print(f"CAP_PROP_FPS                         = {camera.get(cv2.CAP_PROP_FPS                  )}")

    # Set auto exposure to off (mode 1) or manual mode (mode 0.75)
    # Note: This might not work with all cameras
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
    auto_exposure = 1
    auto_exposure_values = [-1, 0, 0.25, 0.75, 1]

    # Set the properties
    camera.set(cv2.CAP_PROP_BRIGHTNESS,  0)      # min: 0, max: 255, default = 0.0
    camera.set(cv2.CAP_PROP_CONTRAST, 32.0)      # min: 0, max: 255, default = 32.0
    camera.set(cv2.CAP_PROP_SATURATION, 75)     # min: 0, max: 255, default = 60.0
    camera.set(cv2.CAP_PROP_HUE, 0)             # min: 0, max: 180, default = 0.0
    camera.set(cv2.CAP_PROP_GAIN, 0.0)          # min: 0, max: 255, default = 0.0
    camera.set(cv2.CAP_PROP_EXPOSURE, -5)       # min: -7, max: -1, default = -5.0

    # Create the sliders
    cv2.createTrackbar("Brightness", "Sliders", int(camera.get(cv2.CAP_PROP_BRIGHTNESS)), 255, lambda x: camera.set(cv2.CAP_PROP_BRIGHTNESS, x))
    cv2.createTrackbar("Contrast", "Sliders", int(camera.get(cv2.CAP_PROP_CONTRAST)), 255, lambda x: camera.set(cv2.CAP_PROP_CONTRAST, x))
    cv2.createTrackbar("Saturation", "Sliders", int(camera.get(cv2.CAP_PROP_SATURATION)), 255, lambda x: camera.set(cv2.CAP_PROP_SATURATION, x))
    cv2.createTrackbar("Hue", "Sliders", int(camera.get(cv2.CAP_PROP_HUE)), 180, lambda x: camera.set(cv2.CAP_PROP_HUE, x))
    cv2.createTrackbar("Gain", "Sliders", int(camera.get(cv2.CAP_PROP_GAIN)), 255, lambda x: camera.set(cv2.CAP_PROP_GAIN, x))

    # Create a list of Exposure values
    exposure_values = [-7, -6, -5, -4, -3, -2, -1]

    # Create a slider for Exposure
    cv2.createTrackbar("Exposure", "Sliders", exposure_values.index(int(camera.get(cv2.CAP_PROP_EXPOSURE))), len(exposure_values)-1, lambda x: camera.set(cv2.CAP_PROP_EXPOSURE, exposure_values[x]))

    while True:
        retval, im = camera.read()

        cv2.imshow("Viewer", im)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        if k == ord('a') or k == ord('A'):
            auto_exposure = auto_exposure - 1
            if auto_exposure == -1:
                auto_exposure = 4
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_values[auto_exposure])

            print("Changing auto exposure")
            print(f"CAP_PROP_BRIGHTNESS                  = {camera.get(cv2.CAP_PROP_BRIGHTNESS           )}")
            print(f"CAP_PROP_CONTRAST                    = {camera.get(cv2.CAP_PROP_CONTRAST             )}")
            print(f"CAP_PROP_SATURATION                  = {camera.get(cv2.CAP_PROP_SATURATION           )}")
            print(f"CAP_PROP_HUE                         = {camera.get(cv2.CAP_PROP_HUE                  )}")
            print(f"CAP_PROP_GAIN                        = {camera.get(cv2.CAP_PROP_GAIN                 )}")
            print(f"CAP_PROP_EXPOSURE                    = {camera.get(cv2.CAP_PROP_EXPOSURE             )}")
            print(f"CAP_PROP_AUTO_EXPOSURE               = {camera.get(cv2.CAP_PROP_AUTO_EXPOSURE        )} = {auto_exposure_values[auto_exposure]}")


    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--CameraID", required = False, help = "Specify Camera ID: python main_config_cap.py --CameraID 0")
    args = vars(ap.parse_args())

    # Default configurations
    CameraID = 0;

    if args["CameraID"]:
        CameraID = int(args["CameraID"])

    main(CameraID)