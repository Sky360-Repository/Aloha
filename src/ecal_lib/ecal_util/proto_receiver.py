# \copyright    Sky360.org
#
# \brief        Subscribes to a topic and receives the messages
#               defined by the proto buffer.
#
# ************************************************************************

import sys
import importlib
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber


class ProtoReceiver:
    # Initialization
    # INPUT:
    # - channel_name: Name of the Channel.
    # - message_name: Name of the Message.
    # - proto_file:   Path and filename of the proto buffer
    def __init__(self, channel_name, message_name, proto_file):
        self.message_name = message_name
        self.proto_file = proto_file
        self.process_name = channel_name + "_subscriber"
        self.start()
        self.subscriber = self.start_subscriber(channel_name, message_name, proto_file)

    @staticmethod
    def get_proto(message_name, proto_file):
        proto = importlib.import_module(proto_file)
        return getattr(proto, message_name)

    def start_subscriber(self, channel_name, message_name, proto_file):
        message_type = self.get_proto(message_name, proto_file)
        return ProtoSubscriber(channel_name, message_type)

    def start(self):
        ecal_core.initialize(sys.argv, self.process_name)
        ecal_core.set_process_state(1, 1, "ECAL running OK")

    def receive(self, wait_time):
        ret, message, time_stamp = self.subscriber.receive(wait_time)
        return ret != 0, message, time_stamp
