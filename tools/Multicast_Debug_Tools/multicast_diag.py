# \copyright    Sky360.org
#
# \brief        Script to test multicast using socket.
#
# ********************************************************
#
# How to Use It
# 1. Run receiver on one or more devices:
#
# python3 multicast_diag.py recv --iface 192.168.1.231
#
# Replace 192.168.1.101 with the local IP of the receiver device.
#
# 2. Run sender on another device:
#
# python3 multicast_diag.py send --iface 192.168.1.197
#
# Again, replace with the sender's local IP.
#
# You should see messages arriving at the receivers.

import socket
import struct
import sys
import argparse
import threading
import time

MULTICAST_GROUP = "239.0.0.1"
PORT = 14000
DEFAULT_TTL = 2
MESSAGE = b"Multicast test from %s"

def sender(interface_ip, ttl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Set TTL
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    # Bind to the specific interface
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(interface_ip))

    print(f"[Sender] Sending on {interface_ip} to {MULTICAST_GROUP}:{PORT} with TTL={ttl}")
    while True:
        msg = MESSAGE % interface_ip.encode()
        sock.sendto(msg, (MULTICAST_GROUP, PORT))
        time.sleep(1)

def receiver(interface_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(('', PORT))
    except Exception as e:
        print(f"[Receiver] Failed to bind socket: {e}")
        sys.exit(1)

    mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(interface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print(f"[Receiver] Listening on {interface_ip} for multicast group {MULTICAST_GROUP}:{PORT}")
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"[Receiver] Received: {data.decode()} from {addr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multicast Diagnostic Tool")
    parser.add_argument("mode", choices=["send", "recv"], help="Mode: send or recv")
    parser.add_argument("--iface", required=True, help="Interface IP address to bind to")
    parser.add_argument('--ttl', type=int, default=DEFAULT_TTL, help='Multicast TTL value')
    args = parser.parse_args()

    if args.mode == "send":
        sender(args.iface, args.ttl)
    elif args.mode == "recv":
        receiver(args.iface)
