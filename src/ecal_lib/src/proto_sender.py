# \copyright    Sky360.org
#
# \brief        Subscribes to a topic and broadcasts the messages
#               defined by the proto buffer.
#
# ************************************************************************

import importlib

from ecal.core.publisher import ProtoPublisher
from ecal_lib.utils.set_process_name import set_process_name
from ecal_lib.utils.credentials import Credentials


class ProtoSender:
    """
    Expects a wrapper message of the form:
        message Wrapper {
            Header header = 1;
            Payload data  = 2;
        }

    - get_message_type() returns a Payload instance (wrapper.data type)
    - send(payload) wraps it with header and publishes Wrapper
    """

    def __init__(self, channel_name: str, wrapper_message_name: str, proto_module: str):
        self.channel_name = channel_name
        self.wrapper_message_name = wrapper_message_name
        self.proto_module = proto_module

        self.process_name = f"{channel_name}_Publisher"
        set_process_name(self.process_name)

        self._wrapper_type = self._load_msg_type(wrapper_message_name, proto_module)

        sample = self._wrapper_type()
        if not (hasattr(sample, "header") and hasattr(sample, "data")):
            raise TypeError(
                f"{wrapper_message_name} must be a wrapper message with fields 'header' and 'data'. "
                f"Got fields: {[f.name for f in sample.DESCRIPTOR.fields]}"
            )

        self._header_type = sample.header.__class__
        self._data_type = sample.data.__class__

        # Credentials are now fully encapsulated
        self._credentials = Credentials()

        self._header = self._build_header()
        self.publisher = ProtoPublisher(channel_name, self._wrapper_type)

    @staticmethod
    def _load_msg_type(message_name: str, proto_module: str):
        module = importlib.import_module(proto_module)
        return getattr(module, message_name)

    def get_message_type(self):
        return self._data_type()

    def send(self, data_message):
        wrapper = self._wrapper_type()
        wrapper.header.CopyFrom(self._header)
        wrapper.data.CopyFrom(data_message)
        self.publisher.send(wrapper)

    def _build_header(self):
        header = self._header_type()
        self._credentials.apply_to_header(header)
        return header

