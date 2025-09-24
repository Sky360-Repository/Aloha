# \copyright    Sky360.org
#
# \brief        Subscribes to a topic and broadcasts the messages
#               defined by the proto buffer.
#
# ************************************************************************

import sys
import importlib
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher


class ProtoSender:
    # Initialization
    # INPUT:
    # - channel_name: Name of the Channel.
    # - message_name: Name of the message.
    # - proto_file:   Path and filename of the proto buffer
    def __init__(self, channel_name, message_name, proto_file):
        self.message_name = message_name
        self.proto_file = proto_file
        self.process_name = channel_name + "_Publisher"
        self.start()
        self.publisher = self.start_publisher(channel_name, message_name, proto_file)

    @staticmethod
    def get_proto(message_name, proto_file):
        proto = importlib.import_module(proto_file)
        return getattr(proto, message_name)

    def get_message_type(self):
        return self.get_proto(self.message_name, self.proto_file)()

    def start_publisher(self, channel_name, message_name, proto_file):
        message_type = self.get_proto(message_name, proto_file)
        return ProtoPublisher(channel_name, message_type)

    def start(self):
        ecal_core.initialize(sys.argv, self.process_name)
        ecal_core.set_process_state(1, 1, "ECAL running OK")

    def send(self, input_message):
        self.publisher.send(input_message)
