# \copyright    Sky360.org
#
# \brief        Script test multicast using eCAL. 
#
# ********************************************************

import ecal.core.core as ecal_core
import ecal.core.publisher as ecal_pub
import ecal.core.subscriber as ecal_sub
import socket, time

def get_active_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Connect to a public server without sending data
    ip = s.getsockname()[0]
    s.close()
    return ip

# Get local IP dynamically
my_ip = get_active_ip()

# Set eCAL configuration programmatically
#ecal_core.set_config_key("network.interface", my_ip)
#ecal_core.set_config_key("network.ttl", "2")
#ecal_core.set_config_key("network.multicast.group", "239.0.0.1")
#ecal_core.set_config_key("network.broadcast", "false")
#ecal_core.set_config_key("network.enable_loopback", "false")

# Initialize eCAL
ecal_core.initialize([], "ecal_node")

# Set up publisher and subscriber on shared topic
hello_pub = ecal_pub.StringPublisher("ecal_handshake")
hello_sub = ecal_sub.StringSubscriber("ecal_handshake")

known_peers = set()

def handshake_callback(topic, msg, time):
    if msg != my_ip:
        known_peers.add(msg)
        print(f"Discovered peer: {msg}")

hello_sub.set_callback(handshake_callback)

# Periodically announce self and print known peers
while ecal_core.ok():
    hello_pub.send(my_ip)
    print(f"Announced: {my_ip} | Peers: {sorted(known_peers)}")
    time.sleep(2)