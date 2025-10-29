# \copyright    Sky360.org
#
# \brief        Captures from QHYCamera and broadcasts over eCAL.
#
# ************************************************************************

import time
import multiprocessing as mp
import cv2
import argparse

from ecal_lib.ecal_util.proto_receiver import ProtoReceiver
from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.set_process_name import *
from camera_controler.QHYCameraController import QHYCameraController

QHY_CHANNEL_NAME = "QHYCamera"
MESSAGE_NAME = "qhy_camera_image"
PROTO_FILE = "ecal_lib.proto_files.qhy_camera_image_pb2"

STATUS_MESSAGE_NAME = "qhy_camera_status"
STATUS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_status_pb2"
QHY_STATUS_CHANNEL = "QHYCamera_status"

PARAMS_MESSAGE_NAME = "qhy_camera_parameters"
PARAMS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_parameters_pb2"
QHY_PARAMS_CHANNEL = "QHYCamera_parameters"

MAX_RESTART_ATTEMPTS = 3
PULSE_TIMEOUT_SEC = 6
INIT_PULSE_TIMEOUT_SEC = 35

def QHYCamera2ecal(param_queue, status_queue):

    # Set process name
    set_process_name(f"{QHY_CHANNEL_NAME} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(QHY_CHANNEL_NAME, MESSAGE_NAME, PROTO_FILE)

    # QHY Camera Controller
    qhy_camera = QHYCameraController()

    # Infinite loop - main_controller kills the process if it has to terminate
    while True:
        # Capture frame-by-frame
        frame = qhy_camera.get_live_frame()

        # Check if frame is not empty
        if frame is not None:

            # Access the protobuf type definition
            protobuf_message = proto_snd.get_message_type()
            protobuf_message.width = qhy_camera.get_shape()[1]
            protobuf_message.height = qhy_camera.get_shape()[0]
            protobuf_message.bit_per_pixel = 16
            protobuf_message.raw_image = frame.tobytes()
            #protobuf_message.time_stamp = int(time.time() * 1.0e6)
            timestamp_sec = qhy_camera.metadata_buffer[qhy_camera.frame_index][0]
            protobuf_message.time_stamp = int(timestamp_sec * 1.0e6)

            # Send the message to the topic this publisher was created for
            proto_snd.send(protobuf_message)

            # send status message
            status_message = {
                'pulse_time': timestamp_sec,
                'exposure': qhy_camera.get_exposure(),
                'gain': qhy_camera.get_gain(),
                'temperature': qhy_camera.get_temperature()
            }
            status_queue.put(status_message)

            # Maybe pass the not None check in the set function
            while not param_queue.empty():
                config_message = param_queue.get()
                qhy_camera.set_target_brightness(config_message['target_brightness'])
                qhy_camera.set_target_gain(config_message['target_gain'])
                qhy_camera.set_exposure_min(config_message['exposure_min'])
                qhy_camera.set_exposure_max(config_message['exposure_max'])
                qhy_camera.set_gain_min(config_message['gain_min'])
                qhy_camera.set_gain_max(config_message['gain_max'])
                qhy_camera.set_exposure_min_step(config_message['exposure_min_step'])
                qhy_camera.set_gain_min_step(config_message['gain_min_step'])
                qhy_camera.set_compensation_factor(config_message['compensation_factor'])
                qhy_camera.set_target_temperature(config_message['target_temperature'])
                qhy_camera.set_histogram_sampling(config_message['histogram_sampling'])
                qhy_camera.set_histogram_dark_point(config_message['histogram_dark_point'])
                qhy_camera.set_histogram_bright_point(config_message['histogram_bright_point'])

    # Close the capture
    qhy_camera.close()

def main_controller():
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
        proc = mp.Process(target=QHYCamera2ecal, args=(param_queue, status_queue))
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
                status_proto_message = status_proto_snd.get_message_type()
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
            params_proto_received, params_proto_message, _ = params_proto_rec.receive(100)
            if params_proto_received:
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
                    'target_temperature': params_proto_message.target_temperature,
                    'histogram_sampling': params_proto_message.histogram_sampling,
                    'histogram_dark_point': params_proto_message.histogram_dark_point,
                    'histogram_bright_point': params_proto_message.histogram_bright_point,
                }
                param_queue.put(config_message)
                is_reset_qyc_set = params_proto_message.reset_qhy
                is_close_qyc_set = params_proto_message.close_qhy

            # Pulse timeout check
            if is_reset_qyc_set or time.time() - last_pulse_time > pulse_timeout:
                print(f"[Controller] No pulse for {pulse_timeout} sec — restarting Capture")

                # Send the message to the topic
                status_proto_message = status_proto_snd.get_message_type()
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

    print("\n\npython QHYCamera2ecal.py")
    print("\n'ctrl+C' to stop\n")

    main_controller()
