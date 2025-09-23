# \copyright    Sky360.org
#
# \brief        Subscribes to a channel and records the into a hdf5 file.
#
# ************************************************************************

import argparse
from ecal_util.ecal_recorder import EcalRecorder

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python ecal_recorder_node.py --channel_name camera_0")
    ap.add_argument("--message_name", required=False,
                    help="Specify the message_name: python ecal_recorder_node.py --message_name web_camera_image")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file: python ecal_recorder_node.py --proto_file proto_files.web_camera_image_pb2")
    args = vars(ap.parse_args())

    # Default configurations
    channel_name = "camera_0"
    message_name = "web_camera_image"
    proto_file = "proto_files.web_camera_image_pb2"

    print("\n\nDefault usage: python ecal_recorder_node.py --channel_name camera_0 "
          "--message_name web_camera_image --proto_file proto_files.web_camera_image_pb2\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["message_name"]:
        message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    recorder = EcalRecorder(channel_name, message_name, proto_file, ecal_meas = '/emmc/ecal_meas/')
    recorder.start_recording()
