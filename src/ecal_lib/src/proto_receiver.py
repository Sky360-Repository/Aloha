import sys
import importlib
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal_lib.utils.set_process_name import set_process_name


class ProtoReceiver:
    def __init__(self, channel_name, wrapper_message_name, proto_file):
        self.wrapper_message_name = wrapper_message_name
        self.proto_file = proto_file

        self.process_name = channel_name + "_subscriber"
        set_process_name(self.process_name)

        self.start()
        self.subscriber = self.start_subscriber(channel_name, wrapper_message_name, proto_file)

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
        ret, wrapper, time_stamp = self.subscriber.receive(wait_time)

        if ret == 0:
            return False, None, time_stamp

        # Ignore header completely
        data = wrapper.data

        return True, data, time_stamp
