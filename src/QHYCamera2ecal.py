# \copyright    Sky360.org
#
# \brief        Captures from QHYCamera and broadcasts over eCAL.
#
# ************************************************************************

import time
import cv2
import argparse

from ecal.ecal_util.proto_sender import ProtoSender
from ecal.ecal_util.jpeg_compression import convert_image_to_byte_array
from ecal.ecal_util.set_process_name import *
from camera_controler.QHYCameraController import QHYCameraController

message_name = "qhy_camera_image"
proto_file = "proto_files.qhy_camera_image_pb2"

def QHYCamera2ecal(channel_name, debug_view):

    if debug_view:
        cv2.namedWindow(channel_name + " Broadcast", cv2.WINDOW_NORMAL)

    # Set process name
    set_process_name(f"{channel_name} Broadcast")

    # Proto sender
    proto_snd = ProtoSender(channel_name, message_name, proto_file)

    # TODO: Add receiver for camera controls and set functions in QHYCameraController

    # QHY Camera Controller
    qhy_camera = QHYCameraController()

    # Infinite loop (using ecal_core.ok() will enable us to shutdown
    # the process from another application
    while True:
        # Capture frame-by-frame
        img = qhy_camera.get_live_frame()

        # Check if frame is not empty
        if img is None:
            print(f"Camera not available")
            break
        else:
            # Access the protobuf type definition
            # TODO: add debug info (get functions in QHYCameraController and fields in protobuf)
            protobuf_message = proto_snd.message
            protobuf_message.width = qhy_camera.get_shape()[1]
            protobuf_message.height = qhy_camera.get_shape()[0]
            protobuf_message.bit_per_pixel = 16
            protobuf_message.raw_image = img.tobytes()
            protobuf_message.time_stamp = int(time.time() * 1.0e6)
}
            # Send the message to the topic this publisher was created for
            proto_snd.send(protobuf_message)

        if debug_view:
            # Display the image
            cv2.imshow(channel_name + " Broadcast", img)

            # Esc key to stop
            if cv2.waitKey(1) & 0xFF == 27:
                break

    if debug_view:
        cv2.destroyAllWindows()

    # Close the capture
    qhy_camera.close()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python QHYCamera2ecal.py --channel_name QHYCamera")
    ap.add_argument("--debug_view", required=False,
                    help="Select True for color and False for gray : python QHYCamera2ecal.py --debug_view True")
    args = vars(ap.parse_args())

    # Default configurations
    channel_name = "QHYCamera"
    debug_view = True

    print("\n\nDefault usage: python QHYCamera2ecal.py --channel_name QHYCamera --debug_view True")
    print("\nPress 'Esc' key to stop\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["debug_view"]:
        debug_view = args["debug_view"].lower() == "true"

    QHYCamera2ecal(channel_name, debug_view)
