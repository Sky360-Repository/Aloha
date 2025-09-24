# \copyright    Sky360.org
#
# \brief        Subscribes to a channel and records the into a hdf5 file.
#
# ************************************************************************

import sys
import time
import datetime
import ecal.measurement.hdf5 as ecalhdf5
import ecal.proto.helper as pb_helper
from ecal_util.proto_receiver import ProtoReceiver
from ecal_util.set_process_name import set_process_name


class EcalRecorder:
    def __init__(self, channel_name, message_name, proto_file, max_size_per_file=1000, ecal_meas=None):
        self.channel_name = channel_name
        self.message_name = message_name
        self.proto_file = proto_file
        self.max_size_per_file = max_size_per_file

        # Default recording folder
        ecal_meas = ecal_meas or "C:/ecal_meas/"
        self.output_dir = f"{ecal_meas}{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}_measurement"

        # Set process name
        set_process_name(f"Recorder for {self.channel_name}")

        # Initialize proto receiver
        self.proto_rec = ProtoReceiver(self.channel_name, self.message_name, self.proto_file)

        # Get message type for descriptor
        message_type = self.proto_rec.get_proto(self.message_name, self.proto_file)
        dummy_message = message_type()

        # Channel definition for HDF5
        self.channel = ecalhdf5.Channel(
            self.channel_name,
            pb_helper.get_descriptor_from_type(dummy_message),
            f"proto:{dummy_message.DESCRIPTOR.full_name}"
        )

        # Setup file
        self.file_name = self.channel_name
        self.meas = ecalhdf5.Meas(self.output_dir, 1)
        self.meas.set_file_base_name(self.file_name)
        self.meas.set_max_size_per_file(self.max_size_per_file)
        self.meas.set_channel_description(self.channel.name, self.channel.description)
        self.meas.set_channel_type(self.channel.name, self.channel.type)

    def start_recording(self):
        print(f"Creating {self.output_dir}/{self.file_name}.hdf5 \n")

        timeout_limit = 100
        count_timeout = 0

        while count_timeout < timeout_limit:
            snd_time_stamp = int(time.time() * 1.0e6)
            received, message, rcv_time_stamp = self.proto_rec.receive(100)

            if received:
                count_timeout = 0
                self.meas.add_entry_to_file(
                    message.SerializeToString(),
                    snd_time_stamp,
                    rcv_time_stamp,
                    self.channel_name
                )
            else:
                count_timeout += 1

        if not self.meas.is_ok():
            print("Write error!")
            sys.exit()

        self.meas.close()
        print(f"{self.channel_name} recording terminated\n")
