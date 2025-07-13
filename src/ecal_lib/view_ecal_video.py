# \copyright    Sky360.org
#
# \brief        Subscribes to a video channel and shows the frame.
#
# ************************************************************************

import datetime
import cv2
import argparse

from ecal_util.proto_receiver import ProtoReceiver
from ecal_util.image_message import ImageMessage
from ecal_util.set_process_name import *


def view_ecal_video(channel_name, message_name, proto_file):
    # Default output folder for captured images
    output_dir = "C:/ecal_meas/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    # Set process name
    set_process_name(f"eCAL viewer on {channel_name}")

    proto_rec = ProtoReceiver(channel_name, message_name, proto_file)
    image_msg = ImageMessage(message_name)

    cv2.namedWindow(channel_name + " Viewer", cv2.WINDOW_NORMAL)

    while True:
        if proto_rec.wait_for_message(100):
            image_msg.process_message(proto_rec.message)
            frame = image_msg.get_rgb_image()

            # Overlay the time stamp
            frame_overlay = cv2.putText(frame, "time_stamp: " + str(image_msg.get_time_stamp()), (30, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                                        cv2.LINE_AA)

            # Overlay the channel Name
            frame_overlay = cv2.putText(frame_overlay, "channel: " + channel_name, (30, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                                        2, cv2.LINE_AA)

            # Display the image
            cv2.imshow(channel_name + " Viewer", frame_overlay)

        k = cv2.waitKey(1)
        # Esc key to stop
        if k == 27:
            break
        # Space to capture the frame
        elif (k == 32) and (image_msg.get_rgb_image() is not None):
            # Save the resulting frame
            cv2.imwrite(output_dir + '_' + message_name + '_' + str(image_msg.get_time_stamp()) + '.bmp',
                        image_msg.get_rgb_image())

    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python view_ecal_video.py --channel_name camera_0")
    ap.add_argument("--message_name", required=False,
                    help="Specify the message_name: python view_ecal_video.py --message_name web_camera_image")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file: python view_ecal_video.py --proto_file proto_files.web_camera_image_pb2")
    args = vars(ap.parse_args())

    # Default configurations
    channel_name = "camera_0"
    message_name = "web_camera_image"
    proto_file = "proto_files.web_camera_image_pb2"

    print("\n\nDefault usage: python view_ecal_video.py --channel_name camera_0 "
          "--message_name web_camera_image --proto_file proto_files.web_camera_image_pb2")
    print("\nPress 'Esc' key to stop")
    print("Press 'Space' key to save the frame\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["message_name"]:
        message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    view_ecal_video(channel_name, message_name, proto_file)
