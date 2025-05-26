# \copyright    Sky360.org
#
# \brief        Captures video from a webcam and  broadcasts over eCAL.
#
# ************************************************************************

import time
import cv2
import argparse

from ecal_util.proto_sender import ProtoSender
from ecal_util.jpeg_compression import convert_image_to_byte_array
from ecal_util.set_process_name import *


def webcam2ecal(camera_id, channel_name, message_name, proto_file):

    # jpeg compression rate in [0, 100]
    # where 100% is the best quality
    # To record raw set jpeg_quality = -1
    jpeg_quality = 100

    # Open the webcam
    # Windows backends
    # - CAP_DSHOW (DirectShow)
    # - CAP_MSMF (Microsoft Media Foundation)
    # - CAP_VFW (Video For Windows)
    #
    # Backend available on Orang is V4L2
    # Default is V4L2 but doesn't allow to change parameters
    cap = cv2.VideoCapture(camera_id)

    cv2.namedWindow(channel_name + " Broadcast", cv2.WINDOW_NORMAL)

    # Resolution Problem
    # Windows sets default parameters that are high frame rate and low resolution
    # Setting to a high resolution and uncompressed
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Set auto exposure to off (mode 1) or manual mode (mode 0.75)
    # Note: This might not work with all cameras
    # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # Set the properties
    #camera.set(cv2.CAP_PROP_BRIGHTNESS,  0)     # min: 0, max: 255, default = 0.0
    #camera.set(cv2.CAP_PROP_CONTRAST, 32.0)     # min: 0, max: 255, default = 32.0
    #camera.set(cv2.CAP_PROP_SATURATION, 75)     # min: 0, max: 255, default = 60.0
    #camera.set(cv2.CAP_PROP_HUE, 0)             # min: 0, max: 180, default = 0.0
    #camera.set(cv2.CAP_PROP_GAIN, 0.0)          # min: 0, max: 255, default = 0.0
    #camera.set(cv2.CAP_PROP_EXPOSURE, -5)       # min: -13, max: -1, default = -5.0

    print(f"max_width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"max_height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

    # Set process name
    set_process_name(f"{channel_name} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(channel_name, message_name, proto_file)

    # Time in Microseconds
    corr_time = int(time.time() * 1.0e6)
    avg_run_time = 1

    # Infinite loop (using ecal_core.ok() will enable us to shutdown
    # the process from another application
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is not empty
        if not ret:
            print(f"Camera {camera_id} not available")
            break

        # Access the protobuf type definition
        protobuf_message = proto_snd.message
        protobuf_message.width = frame.shape[1]
        protobuf_message.height = frame.shape[0]
        if jpeg_quality == -1:
            protobuf_message.raw_image = frame.tobytes()
            protobuf_message.jpeg_size = 0
        else:
            byte_array = convert_image_to_byte_array(frame, jpeg_quality)
            protobuf_message.jpeg_data = byte_array
            protobuf_message.jpeg_size = len(byte_array)

        protobuf_message.sensor_id = camera_id
        protobuf_message.time_stamp = int(time.time() * 1.0e6)

        # Send the message to the topic this publisher was created for
        proto_snd.send(protobuf_message)

        # Calculate run frequency
        prev_time = corr_time
        corr_time = int(time.time() * 1.0e6)
        avg_run_time = 0.9 * avg_run_time + 0.1 * (corr_time - prev_time)
        run_freq = 1.0e6 / avg_run_time

        # Display time stamp
        frame = cv2.putText(frame, "run_freq: " + str(run_freq), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Image Size: [" + str(frame.shape[1]) + ", " + str(frame.shape[0]) + "]",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            2, cv2.LINE_AA)

        # Display the image
        cv2.imshow(channel_name + " Broadcast", frame)

        # Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python webcam2ecal.py --channel_name camera_0")
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python webcam2ecal.py --camera_id 0")
    ap.add_argument("--message_name", required=False,
                    help="Specify the message_name: python webcam2ecal.py --message_name web_camera_image")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file: python webcam2ecal.py --proto_file proto_files.web_camera_image_pb2")
    args = vars(ap.parse_args())

    # Default configurations
    camera_id = 0
    channel_name = "camera_" + str(camera_id)
    message_name = "web_camera_image"
    proto_file = "proto_files.web_camera_image_pb2"

    print("\n\nDefault usage: python webcam2ecal.py --camera_id 0 --channel_name camera_0 "
          "--message_name web_camera_image --proto_file proto_files.web_camera_image_pb2")
    print("\nPress 'Esc' key to stop\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["message_name"]:
        message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    webcam2ecal(camera_id, channel_name, message_name, proto_file)
