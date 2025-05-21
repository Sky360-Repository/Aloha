# \copyright    Sky360.org
#
# \brief        Subscribes to a channel and records the into a hdf5 file.
#
# ************************************************************************

import time
import datetime
import argparse
import ecal.measurement.hdf5 as ecalhdf5
import ecal.proto.helper as pb_helper

from ecal_util.proto_receiver import ProtoReceiver
from ecal_util.set_process_name import *


def ecal_recorder(channel_name, message_name, proto_file):

    # Max file size Mb.
    # eCAL create multiple files per topic, each file will have this size.
    max_size_per_file = 500

    # Default recording folder
    output_dir = ("C:/ecal_meas/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3] +
                  "_measurement")

    # Set process name
    set_process_name(f"Recorder for {channel_name}")

    proto_rec = ProtoReceiver(channel_name, message_name, proto_file)

    # Channel definition for hdf5
    channel = ecalhdf5.Channel(channel_name, pb_helper.get_descriptor_from_type(proto_rec.message),
                               "proto:" + proto_rec.message.DESCRIPTOR.full_name)

    # File setup for hdf5
    file_name = channel_name
    meas = ecalhdf5.Meas(output_dir, 1)
    meas.set_file_base_name(file_name)
    meas.set_max_size_per_file(max_size_per_file)

    meas.set_channel_description(channel.name, channel.description)
    meas.set_channel_type(channel.name, channel.type)

    print(f"Creating {output_dir}/{file_name}.hdf5 \n")

    # Time stamp is in microseconds
    count_timeout = 0
    while True:
        snd_time_stamp = int(time.time() * 1.0e6)
        if proto_rec.wait_for_message(100):
            count_timeout = 0
            rcv_time_stamp = int(time.time() * 1.0e6)
            meas.add_entry_to_file(proto_rec.message.SerializeToString(), snd_time_stamp, rcv_time_stamp, channel_name)
        else:
            count_timeout = count_timeout + 1
        if count_timeout == 100:
            break

    if not meas.is_ok():
        print("Write error!")
        sys.exit()

    meas.close()
    print(f"{channel_name} recording terminated\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python ecal_recorder.py --channel_name camera_0")
    ap.add_argument("--message_name", required=False,
                    help="Specify the message_name: python ecal_recorder.py --message_name web_camera_image")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file: python ecal_recorder.py --proto_file proto_files.web_camera_image_pb2")
    args = vars(ap.parse_args())

    # Default configurations
    channel_name = "camera_0"
    message_name = "web_camera_image"
    proto_file = "proto_files.web_camera_image_pb2"

    print("\n\nDefault usage: python ecal_recorder.py --channel_name camera_0 "
          "--message_name web_camera_image --proto_file proto_files.web_camera_image_pb2\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["message_name"]:
        message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    ecal_recorder(channel_name, message_name, proto_file)
