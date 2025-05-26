# \copyright    Sky360.org
#
# \brief        Test multicast using socket.
#
# ********************************************************
#
# python pub_sub_handshake_with_socket.py --ttl 2

import socket
import struct
import time
import threading
import argparse

MULTICAST_GROUP = "239.0.0.1"
PORT = 14000
DEFAULT_TTL = 2

known_peers = {
    "192.168.1.231",
    "192.168.1.101"
}

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Connect to a public server without sending data
    ip = s.getsockname()[0]
    s.close()
    return ip

def sender(multicast_group, port, ttl, local_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    ttl_bin = struct.pack('b', ttl)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl_bin)

    while True:
        message = f"HELLO_FROM_{local_ip}"
        sock.sendto(message.encode(), (multicast_group, port))
        print(f"    [SEND] {message}")
        time.sleep(1)

def receiver(multicast_group, port, local_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', port))

    mreq = struct.pack('4s4s', socket.inet_aton(multicast_group), socket.inet_aton(local_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print(f"[RECV] Listening for messages on {multicast_group}:{port} (local IP: {local_ip})")

    while True:
        data, addr = sock.recvfrom(1024)
        peer_ip = addr[0]
        if peer_ip != local_ip:
            if peer_ip in known_peers:
                print(f"[RECV] From {peer_ip} and is a known peer")
            else:
                print(f"[RECV] From {peer_ip} and is unknown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multicast Pub-Sub Handshake Test")
    parser.add_argument('--ttl', type=int, default=DEFAULT_TTL, help='Multicast TTL value')
    args = parser.parse_args()

    local_ip = get_local_ip()
    print(f"[INFO] Local IP: {local_ip}")

    t_send = threading.Thread(target=sender, args=(MULTICAST_GROUP, PORT, args.ttl, local_ip), daemon=True)
    t_recv = threading.Thread(target=receiver, args=(MULTICAST_GROUP, PORT, local_ip), daemon=True)

    t_send.start()
    t_recv.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
