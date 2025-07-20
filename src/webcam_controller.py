# \copyright    Sky360.org
#
# \brief        Captures video from a webcam and  broadcasts over eCAL.
#
# ************************************************************************

import time
import multiprocessing as mp
import cv2
import argparse

from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.jpeg_compression import convert_image_to_byte_array
from ecal_lib.ecal_util.set_process_name import *

MAX_RESTART_ATTEMPTS = 3
PULSE_TIMEOUT_SEC = 5
INIT_PULSE_TIMEOUT_SEC = 30

# jpeg compression rate in [0, 100]
# where 100% is the best quality
# To record raw set JPEG_QUALITY = -1
JPEG_QUALITY = 100

def webcam2ecal(init_params, param_queue, status_queue):

    # Init Parameters
    camera_id = init_params['camera_id']
    channel_name = init_params['channel_name']
    message_name = init_params['message_name']
    proto_file = init_params['proto_file']

    cv2.namedWindow(channel_name + " Broadcast", cv2.WINDOW_NORMAL)

    # Open the webcam
    cap = cv2.VideoCapture(camera_id)

    # Resolution Problem
    # Windows sets default parameters that are high frame rate and low resolution
    # Setting to a high resolution and uncompressed
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Set process name
    set_process_name(f"{channel_name} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(channel_name, message_name, proto_file)

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
        if JPEG_QUALITY == -1:
            protobuf_message.raw_image = frame.tobytes()
            protobuf_message.jpeg_size = 0
        else:
            byte_array = convert_image_to_byte_array(frame, JPEG_QUALITY)
            protobuf_message.jpeg_data = byte_array
            protobuf_message.jpeg_size = len(byte_array)

        protobuf_message.sensor_id = camera_id
        protobuf_message.time_stamp = int(time.time() * 1.0e6)

        # Send the message to the topic this publisher was created for
        proto_snd.send(protobuf_message)

        # send status message
        status_message = {
            'pulse_time': time.time(),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS)
        }
        status_queue.put(status_message)

        #while not param_queue.empty():
        #    config_message = param_queue.get()
        #    cap.set(cv2.CAP_PROP_BRIGHTNESS,  config_message['brightness'])

        # Display the image
        cv2.imshow(channel_name + " Broadcast", frame)

        # Esc key to stop
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print(f"[webcam2ecal] Blocked ...")
            while True:
                time.sleep(1)

    # When everything done, release the capture
    cv2.destroyAllWindows()

    # When everything done, release the capture
    cap.release()

def webcam_controller(camera_id, channel_name, message_name, proto_file):
    param_queue = mp.Queue()
    status_queue = mp.Queue()
    restart_attempts = 0

    # init parameters
    init_params = {
        'camera_id': camera_id,
        'channel_name': channel_name,
        'message_name': message_name,
        'proto_file': proto_file
    }

    def start_capture():
        print("[Controller] start_capture")
        proc = mp.Process(target=webcam2ecal, args=(init_params, param_queue, status_queue))
        proc.start()
        return proc

    capture_proc = start_capture()

    # param_queue.put(camera_params)
    last_pulse_time = time.time()
    # initial pulse timeout in seconds
    pulse_timeout = INIT_PULSE_TIMEOUT_SEC
    while True:
        try:
            # Get status message
            while not status_queue.empty():
                status_message = status_queue.get()
                # publish_status_through_ecal()
                last_pulse_time = status_message['pulse_time']
                pulse_timeout = PULSE_TIMEOUT_SEC
                restart_attempts = 0

            #if get_param_from_ecal:
            #    param_queue.put(updated_config)
            print(f"[Controller] {time.time() - last_pulse_time} > {pulse_timeout}")

            # Pulse timeout check
            if time.time() - last_pulse_time > pulse_timeout:
                print(f"[Controller] No pulse for {pulse_timeout} sec — restarting Capture")
                capture_proc.terminate()
                capture_proc.join()

                restart_attempts += 1
                if restart_attempts >= MAX_RESTART_ATTEMPTS:
                    print("[Controller] Max restart attempts reached — manual intervention required")
                    break

                capture_proc = start_capture()
                last_pulse_time = time.time()
                pulse_timeout = INIT_PULSE_TIMEOUT_SEC
                # Remove: this parameter is to test restart attempts
                init_params['camera_id'] = 9

            time.sleep(1)

        except KeyboardInterrupt:
            print("[Controller] Manual shutdown")
            capture_proc.terminate()
            break

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
    proto_file = "ecal_lib.proto_files.web_camera_image_pb2"

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

    webcam_controller(camera_id, channel_name, message_name, proto_file)
