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
        self.process_name = channel_name + "_Publisher"
        self.start()
        self.publisher = None
        self.ret = int()
        self.message = self.start_publisher(channel_name, message_name, proto_file)()
        self.message_was_sent = False

    @staticmethod
    def get_proto(message_name, proto_file):
        proto = importlib.import_module(proto_file)
        return getattr(proto, message_name)

    def start_publisher(self, channel_name, message_name, proto_file):
        message = self.get_proto(message_name, proto_file)
        self.publisher = ProtoPublisher(channel_name, message)
        self.message_was_sent = False
        return message

    def start(self):
        ecal_core.initialize(sys.argv, self.process_name)
        ecal_core.set_process_state(1, 1, "ECAL running OK")

    def send(self, input_message):
        self.publisher.send(input_message)
