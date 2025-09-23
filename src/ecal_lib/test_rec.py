# \copyright    Sky360.org
#
# \brief        eCAL receiver.
#
# ************************************************************************

import socket
import ecal.core.core as ecal_core
from ecal_util.proto_receiver import ProtoReceiver
from ecal_util.set_process_name import set_process_name


def get_active_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Connect to a public server without sending data
    ip = s.getsockname()[0]
    s.close()
    return ip


# Get local IP
my_ip = get_active_ip()

channel_name = "test massager"
message_name = "test_msg"
proto_file = "proto_files.test_proto_pb2"

if __name__ == "__main__":
    # Set process name
    set_process_name(f"Receiver test from {my_ip}")

    proto_rec = ProtoReceiver(channel_name, message_name, proto_file)

    # ecal_core.ok() for a gracefully shutdown
    while ecal_core.ok():
        received, message, _ = proto_rec.receive(100)
        if received:
            print("Message from {} at {} : {}".format(message.name,
                                                      message.ip,
                                                      message.msg))

            if (message.ip == my_ip and (message.msg == 'q' or message.msg == 'Q')):
                break
