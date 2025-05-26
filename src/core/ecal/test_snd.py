# \copyright    Sky360.org
#
# \brief        eCAL sender.
#
# ************************************************************************

import socket
from ecal_util.proto_sender import ProtoSender
from ecal_util.set_process_name import set_process_name

def get_active_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Connect to a public server without sending data
    ip = s.getsockname()[0]
    s.close()
    return ip

# Get local IP
my_ip = get_active_ip()

channel_name = "test messager"
message_name = "test_msg"
proto_file = "proto_files.test_proto_pb2"

if __name__ == "__main__":

    name = input("Please enter your name:")

    # Set process name
    set_process_name(f"Sender test from {my_ip}")

    proto_snd = ProtoSender(channel_name, message_name, proto_file)

    print("Type 'q' exit")
    while True:
        input_message = input("Type the message you want to send:")

        # Access the Message type definition
        protobuf_message = proto_snd.message
        protobuf_message.name = name
        protobuf_message.ip = my_ip
        protobuf_message.msg = input_message

        # actually send the message to the topic this publisher was created for
        proto_snd.send(protobuf_message)

        if input_message == 'q' or input_message == 'Q':
            break
