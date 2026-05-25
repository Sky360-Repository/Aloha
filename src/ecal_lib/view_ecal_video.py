# \copyright    Sky360.org
#
# \brief        Subscribes to a video channel and shows the frame.
#
# ************************************************************************

import datetime
import cv2
import argparse

from ecal_lib.src.proto_receiver import ProtoReceiver
from ecal_lib.utils.image_message import ImageMessage


def view_ecal_video(channel_name, message_name, proto_file):
    # Default output folder for captured images
    output_dir = "C:/ecal_meas/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    proto_rec = ProtoReceiver(channel_name, message_name, proto_file)

    # ImageMessage should work on the message_name type
    image_msg = ImageMessage(message_name)

    cv2.namedWindow(channel_name + " Viewer", cv2.WINDOW_NORMAL)

    while True:
        received, message, _ = proto_rec.receive(100)

        if received:
            image_msg.process_message(message)
            frame = image_msg.get_rgb_image()

            if frame is not None:
                cv2.imshow(channel_name + " Viewer", frame)

        k = cv2.waitKey(1)

        # Esc key to stop
        if k == 27:
            break

        # Space to capture the frame
        elif (k == 32) and (image_msg.get_rgb_image() is not None):
            cv2.imwrite(
                output_dir + '_' + payload_message_name + '_' + str(image_msg.get_time_stamp()) + '.bmp',
                image_msg.get_rgb_image()
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name")
    ap.add_argument("--message_name", required=False,
                    help="Specify the message name")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file")

    args = vars(ap.parse_args())

    # Example defaults
    channel_name = "camera_0"
    message_name = "web_camera_image"
    proto_file = "ecal_lib.proto_files.web_camera_image_pb2"

    print("\nDefault usage:")
    print("python view_ecal_video.py "
          "--channel_name camera_0 "
          "--message_name web_camera_image "
          "--proto_file ecal_lib.proto_files.web_camera_image_pb2")
    print("\nPress 'Esc' key to stop")
    print("Press 'Space' key to save the frame\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["message_name"]:
        message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    view_ecal_video(channel_name, message_name, proto_file)
