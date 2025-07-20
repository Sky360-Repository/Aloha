# \copyright    Sky360.org
#
# \brief        Captures from QHYCamera and broadcasts over eCAL.
#
# ************************************************************************

import time
import multiprocessing as mp
import argparse

from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.set_process_name import *
from camera_controler.QHYCameraController import QHYCameraController

MAX_RESTART_ATTEMPTS = 3
PULSE_TIMEOUT_SEC = 5
INIT_PULSE_TIMEOUT_SEC = 30

MESSAGE_NAME = "qhy_camera_image"
PROTO_FILE = "ecal_lib.proto_files.qhy_camera_image_pb2"

def QHYCamera2ecal(channel_name, param_queue, status_queue):

    # Set process name
    set_process_name(f"{channel_name} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(channel_name, MESSAGE_NAME, PROTO_FILE)

    # QHY Camera Controller
    qhy_camera = QHYCameraController()

    # Infinite loop - main_controller kills the process if it has to terminate
    while True:
        # Capture frame-by-frame
        img = qhy_camera.get_live_frame()

        # Check if frame is not empty
        if img is None:
            print(f"Camera not available")
            break
        else:
            protobuf_message = proto_snd.message
            protobuf_message.width = qhy_camera.get_shape()[1]
            protobuf_message.height = qhy_camera.get_shape()[0]
            protobuf_message.bit_per_pixel = 16
            protobuf_message.raw_image = img.tobytes()
            protobuf_message.time_stamp = int(time.time() * 1.0e6)

            # Send the message to the topic this publisher was created for
            proto_snd.send(protobuf_message)

            # send status message
            status_message = {
                'pulse_time': time.time(),
                # TODO: add these get functions in QHYCameraController
                # 'timestamp': qhy_camera.get_timestamp()
                # 'exposure': qhy_camera.exposure()
                # 'gain': qhy_camera.gain()
                # 'temperature': qhy_camera.temperature()
            }
            status_queue.put(status_message)

            # TODO: add these set functions in QHYCameraController
            # Maybe pass the not None check in the set function
            # while not param_queue.empty():
            #     config_message = param_queue.get()
            #     if (target_brightness := config_message['target_brightness']) is not None:
            #         qhy_camera.set_target_brightness(target_brightness)
            #     if (target_gain := config_message['target_gain']) is not None:
            #         qhy_camera.set_target_gain(target_gain)
            #     if (exposure_min := config_message['exposure_min']) is not None:
            #         qhy_camera.set_exposure_min(exposure_min)
            #     if (exposure_max := config_message['exposure_max']) is not None:
            #         qhy_camera.set_exposure_max(exposure_max)
            #     if (gain_min := config_message.get('gain_min') is not None:
            #         qhy_camera.set_gain_min(gain_min)
            #     if (gain_max := config_message.get('gain_max') is not None:
            #         qhy_camera.set_gain_max(gain_max)
            #     if (exposure_min_step := config_message.get('exposure_min_step') is not None:
            #         qhy_camera.set_exposure_min_step(exposure_min_step)
            #     if (gain_min_step := config_message.get('gain_min_step') is not None:
            #         qhy_camera.set_gain_min_step(gain_min_step)
            #     if (compensation_factor := config_message.get('compensation_factor') is not None:
            #         qhy_camera.set_compensation_factor(compensation_factor)
            #     if (target_temperature := config_message.get('target_temperature') is not None:
            #         qhy_camera.set_target_temperature(target_temperature)

            # Close the capture
    qhy_camera.close()

def main_controller(channel_name):
    param_queue = mp.Queue()
    status_queue = mp.Queue()
    restart_attempts = 0

    # TODO: Add eCAL sender and receiver for status and parameters

    def start_capture():
        print("[Controller] Start QHY Camera")
        proc = mp.Process(target=QHYCamera2ecal, args=(channel_name, param_queue, status_queue))
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

                # TODO: publish_status_through_ecal()
                # status_message['timestamp']
                # status_message['exposure']
                # status_message['gain']
                # status_message['temperature']

                last_pulse_time = status_message['pulse_time']
                pulse_timeout = PULSE_TIMEOUT_SEC
                restart_attempts = 0

            # TODO: Add receiver from eCAL
            # if get_param_from_ecal:
            #     updated_config.target_brightness = message.target_brightness
            #     updated_config.target_gain = message.target_gain
            #     updated_config.exposure_min = message.exposure_min
            #     updated_config.exposure_max = message.exposure_max
            #     updated_config.gain_min = message.gain_min
            #     updated_config.gain_max = message.gain_max
            #     updated_config.exposure_min_step = message.exposure_min_step
            #     updated_config.gain_min_step = message.gain_min_step
            #     updated_config.compensation_factor = message.compensation_factor
            #     updated_config.target_temperature = message.target_temperature
            #     param_queue.put(updated_config)

            # Pulse timeout check
            if time.time() - last_pulse_time > pulse_timeout:
                print(f"[Controller] No pulse for {pulse_timeout} sec — restarting QHY Camera")
                capture_proc.terminate()
                capture_proc.join()

                restart_attempts += 1
                if restart_attempts >= MAX_RESTART_ATTEMPTS:
                    print("[Controller] Max restart attempts reached — check QHY Camera")
                    break

                capture_proc = start_capture()
                last_pulse_time = time.time()
                pulse_timeout = INIT_PULSE_TIMEOUT_SEC

            time.sleep(1)

        except KeyboardInterrupt:
            print("[Controller] Main QHY controller shutdown")
            capture_proc.terminate()
            break

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python QHYCamera2ecal.py --channel_name QHYCamera")
    args = vars(ap.parse_args())

    # Default configurations
    channel_name = "QHYCamera"
    debug_view = True

    print("\n\nDefault usage: python QHYCamera2ecal.py --channel_name QHYCamera")
    print("\nPress 'Esc' key to stop\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]

    main_controller(channel_name)
