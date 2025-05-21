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
        self.process_name = channel_name + "_subscriber"
        self.start()
        self.time_stamp = 0
        self.message_was_received = False
        self.subscriber = None
        self.ret = int()
        self.message = self.start_subscriber(channel_name, message_name, proto_file)()

    @staticmethod
    def get_proto(message_name, proto_file):
        proto = importlib.import_module(proto_file)
        return getattr(proto, message_name)

    def start_subscriber(self, channel_name, message_name, proto_file):
        message = self.get_proto(message_name, proto_file)
        self.subscriber = ProtoSubscriber(channel_name, message)
        self.message_was_received = False
        return message

    def start(self):
        ecal_core.initialize(sys.argv, self.process_name)
        ecal_core.set_process_state(1, 1, "ECAL running OK")

    def receive(self, wait_time):
        self.ret, self.message, self.time_stamp = self.subscriber.receive(wait_time)
        if self.ret != 0:
            self.message_was_received = True

    def wait_for_message(self, wait_time):
        self.message_was_received = False
        self.receive(wait_time)
        return self.message_was_received
