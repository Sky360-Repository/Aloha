# \copyright    Sky360.org
#
# \brief        Captures video from a webcam and  broadcasts over eCAL.
#
# ************************************************************************

import time
import multiprocessing as mp
import cv2
import argparse

from ecal_lib.ecal_util.proto_receiver import ProtoReceiver
from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.jpeg_compression import convert_image_to_byte_array
from ecal_lib.ecal_util.set_process_name import *

CAMERA_ID = 0
QHY_CHANNEL_NAME = "QHYCamera"
MESSAGE_NAME = "web_camera_image"
PROTO_FILE = "ecal_lib.proto_files.web_camera_image_pb2"

STATUS_MESSAGE_NAME = "qhy_camera_status"
STATUS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_status_pb2"
QHY_STATUS_CHANNEL = "QHYCamera_status"

PARAMS_MESSAGE_NAME = "qhy_camera_parameters"
PARAMS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_parameters_pb2"
QHY_PARAMS_CHANNEL = "QHYCamera_parameters"

MAX_RESTART_ATTEMPTS = 3
PULSE_TIMEOUT_SEC = 5
INIT_PULSE_TIMEOUT_SEC = 30

# jpeg compression rate in [0, 100]
# where 100% is the best quality
# To record raw set JPEG_QUALITY = -1
JPEG_QUALITY = 100

def webcam2ecal(param_queue, status_queue):

    # Set process name
    set_process_name(f"{QHY_CHANNEL_NAME} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(QHY_CHANNEL_NAME, MESSAGE_NAME, PROTO_FILE)

    cv2.namedWindow(QHY_CHANNEL_NAME + " Broadcast", cv2.WINDOW_NORMAL)

    # Open the webcam
    cap = cv2.VideoCapture(CAMERA_ID)

    # Resolution Problem
    # Windows sets default parameters that are high frame rate and low resolution
    # Setting to a high resolution and uncompressed
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Infinite loop - main_controller kills the process if it has to terminate
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is not empty
        if not ret:
            print(f"Camera {CAMERA_ID} not available")
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

        protobuf_message.sensor_id = CAMERA_ID
        protobuf_message.time_stamp = int(time.time() * 1.0e6)

        # Send the message to the topic this publisher was created for
        proto_snd.send(protobuf_message)

        # send status message
        status_message = {
            'pulse_time': time.time(),
            'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
            'gain': cap.get(cv2.CAP_PROP_GAIN),
            'temperature': cap.get(cv2.CAP_PROP_TEMPERATURE )
        }
        status_queue.put(status_message)

        # Maybe pass the not None check in the set function
        while not param_queue.empty():
            config_message = param_queue.get()
            print(f"target_brightness = {config_message['target_brightness']};\n"
                  f"target_gain : {config_message['target_gain']};\n"
                  f"exposure_min : {config_message['exposure_min']};\n"
                  f"exposure_max : {config_message['exposure_max']};\n"
                  f"gain_min : {config_message['gain_min']};\n"
                  f"gain_max : {config_message['gain_max']};\n"
                  f"exposure_min_step : {config_message['exposure_min_step']};\n"
                  f"gain_min_step : {config_message['gain_min_step']};\n"
                  f"compensation_factor : {config_message['compensation_factor']};\n"
                  f"target_temperature : {config_message['target_temperature']};")

        # Display the image
        cv2.imshow(QHY_CHANNEL_NAME + " Broadcast", frame)

        # Esc key to stop
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print(f"[webcam2ecal] QHY crashed ...")
            while True:
                time.sleep(1)

    # Close the capture and close window
    cv2.destroyAllWindows()
    cap.release()

def webcam_controller():
    mp.set_start_method("spawn")
    param_queue = mp.Queue()
    status_queue = mp.Queue()
    restart_attempts = 0

    # Set process name
    set_process_name(f"{QHY_CHANNEL_NAME} Broadcast")

    # Proto sender
    status_proto_snd = ProtoSender(QHY_STATUS_CHANNEL, STATUS_MESSAGE_NAME, STATUS_PROTO_FILE)

    # Proto receiver
    params_proto_rec = ProtoReceiver(QHY_PARAMS_CHANNEL, PARAMS_MESSAGE_NAME, PARAMS_PROTO_FILE)

    def start_capture():
        print("[Controller] Start QHY Camera")
        proc = mp.Process(target=webcam2ecal, args=(param_queue, status_queue))
        proc.start()
        return proc

    capture_proc = start_capture()

    is_reset_qyc_set = False
    is_close_qyc_set = False

    last_pulse_time = time.time()
    # initial pulse timeout in seconds
    pulse_timeout = INIT_PULSE_TIMEOUT_SEC
    while True:
        try:
            # Get status message
            while not status_queue.empty():
                status_message = status_queue.get()

                # Access the protobuf type definition
                status_proto_message = status_proto_snd.message
                status_proto_message.temperature = status_message['temperature']
                status_proto_message.gain = status_message['gain']
                status_proto_message.exposure = status_message['exposure']
                status_proto_message.is_qhy_live = True
                status_proto_message.time_stamp = int(status_message['pulse_time'] * 1.0e6)

                # Send the message to the topic this publisher was created for
                status_proto_snd.send(status_proto_message)

                last_pulse_time = status_message['pulse_time']
                pulse_timeout = PULSE_TIMEOUT_SEC
                restart_attempts = 0

            # Receiver from eCAL
            if params_proto_rec.wait_for_message(100):
                params_proto_message = params_proto_rec.message
                config_message = {
                    'target_brightness': params_proto_message.target_brightness,
                    'target_gain': params_proto_message.target_gain,
                    'exposure_min': params_proto_message.exposure_min,
                    'exposure_max': params_proto_message.exposure_max,
                    'gain_min': params_proto_message.gain_min,
                    'gain_max': params_proto_message.gain_max,
                    'exposure_min_step': params_proto_message.exposure_min_step,
                    'gain_min_step': params_proto_message.gain_min_step,
                    'compensation_factor': params_proto_message.compensation_factor,
                    'target_temperature': params_proto_message.target_temperature
                }
                param_queue.put(config_message)
                is_reset_qyc_set = params_proto_message.reset_qhy
                is_close_qyc_set = params_proto_message.close_qhy

            # Pulse timeout check
            if is_reset_qyc_set or time.time() - last_pulse_time > pulse_timeout:
                print(f"[Controller] No pulse for {pulse_timeout} sec — restarting Capture")

                # Send the message to the topic
                status_proto_message = status_proto_snd.message
                status_proto_message.is_qhy_live = False
                status_proto_message.time_stamp = int(time.time() * 1.0e6)
                status_proto_snd.send(status_proto_message)

                capture_proc.terminate()
                capture_proc.join()

                restart_attempts += 1
                if restart_attempts >= MAX_RESTART_ATTEMPTS:
                    print("[Controller] Max restart attempts reached — check QHY Camera")
                    break

                capture_proc = start_capture()
                last_pulse_time = time.time()
                pulse_timeout = INIT_PULSE_TIMEOUT_SEC
                is_reset_qyc_set = False

            if is_close_qyc_set:
                print(f"[Controller] Closing QHY")
                capture_proc.terminate()
                break

            time.sleep(1)

        except KeyboardInterrupt:
            print("[Controller] Closing QHY")
            capture_proc.terminate()
            break

if __name__ == "__main__":

    print("\n\npython webcam2ecal.py")
    print("\n'ctrl+C' to stop\n")

    webcam_controller()
