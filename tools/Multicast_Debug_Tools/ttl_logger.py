# \copyright    Sky360.org
#
# \brief        Script to listen to any UDP communication.
#
# ********************************************************

import socket
import struct
import time

MULTICAST_GROUP = '239.0.0.1'
PORT = 14000

# Set up receiver socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', PORT))

# Join multicast group
mreq = struct.pack("=4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

print(f"Listening on {MULTICAST_GROUP}:{PORT}...")
while True:
    data, addr = sock.recvfrom(1024)
    print(f"Received from {addr[0]}")