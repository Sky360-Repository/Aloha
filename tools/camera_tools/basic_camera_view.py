# \copyright    Sky360.org
#
# \brief        Basic camera view using OpenCV.
#               Allows to zoom with the mouse click.
#
# ************************************************************************

import cv2
import time
import argparse

cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)

print(cv2.getBuildInformation())

# Initialize a flag for ROI display
roi_display = False

# Define the mouse callback function
def click_and_crop(event, x, y, flags, param):
    global roi_display, frame, roi_size

    if event == cv2.EVENT_LBUTTONDOWN:
        if roi_display:
            roi_display = False
        else:
            roi_display = True
            roi_size = [max(0, y-200), min(frame.shape[0], y+200), max(0, x-300), min(frame.shape[1], x+300)]

def decode_fourcc(fourcc):
    fourcc = int(fourcc) & 0xFFFFFFFF  # Ensure fourcc is a 32-bit unsigned integer
    fourcc_str = [chr((fourcc >> i) & 0xFF) for i in (0, 8, 16, 24)]
    fourcc_str.append('\0')
    return fourcc_str


# Argument for camera ID
ap = argparse.ArgumentParser()
ap.add_argument("--CameraID", required = False, help = "Specify Camera ID: python main_config_cap.py --CameraID 0")
args = vars(ap.parse_args())

# Default configurations
CameraID = 0;

if args["CameraID"]:
    CameraID = int(args["CameraID"])

    # Open the webcam
    # Windows backends
    # - CAP_DSHOW (DirectShow)
    # - CAP_MSMF (Microsoft Media Foundation)
    # - CAP_VFW (Video For Windows)
cap = cv2.VideoCapture(CameraID, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Set auto exposure to off (mode 1) or manual mode (mode 0.75)
# Note: This might not work with all cameras
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
print(f"CAP_PROP_BRIGHTNESS                  = {cap.get(cv2.CAP_PROP_BRIGHTNESS           )}")
print(f"CAP_PROP_CONTRAST                    = {cap.get(cv2.CAP_PROP_CONTRAST             )}")
print(f"CAP_PROP_SATURATION                  = {cap.get(cv2.CAP_PROP_SATURATION           )}")
print(f"CAP_PROP_HUE                         = {cap.get(cv2.CAP_PROP_HUE                  )}")
print(f"CAP_PROP_GAIN                        = {cap.get(cv2.CAP_PROP_GAIN                 )}")
print(f"CAP_PROP_EXPOSURE                    = {cap.get(cv2.CAP_PROP_EXPOSURE             )}")
print(f"CAP_PROP_AUTO_EXPOSURE               = {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE        )}")
print(f"CAP_PROP_FRAME_WIDTH                 = {cap.get(cv2.CAP_PROP_FRAME_WIDTH          )}")
print(f"CAP_PROP_FRAME_HEIGHT                = {cap.get(cv2.CAP_PROP_FRAME_HEIGHT         )}")
print(f"CAP_PROP_FPS                         = {cap.get(cv2.CAP_PROP_FPS                  )}")

## Resolution Problem
# Reset to a high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5448)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print(f"Check configuration: ")
print(f"CameraID = {CameraID}")
print(f"fourcc   = {decode_fourcc(cv2.VideoWriter_fourcc(*'MJPG'))} = {decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))}")
print(f"Config Image = [{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}]; at {cap.get(cv2.CAP_PROP_FPS)} fps")

# Set the mouse callback function for window
cv2.setMouseCallback("Viewer", click_and_crop)

# Time in Microseconds
corr_time = int(time.time() * 1.0e6)
avg_run_time = 1
while True:
    prev_time = corr_time
    corr_time = int(time.time() * 1.0e6)
    avg_run_time = 0.9 * avg_run_time + 0.1 * (corr_time - prev_time)
    run_freq = 1.0e6 / avg_run_time

    # Get the frame from capture
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if roi_display:
        # Define the ROI
        roi = frame[roi_size[0]:roi_size[1], roi_size[2]:roi_size[3]]
        # Replace frame with resize the ROI
        frame = cv2.resize(roi, (1200, 800))

    # Display timestamp
    frame = cv2.putText(frame, "run_freq: " + str(run_freq), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Image Size: [" + str(frame.shape[1]) + ", " + str(frame.shape[0]) + "]", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Viewer", frame)

    k = cv2.waitKey(1)
    if k==27: # Esc key to stop
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
