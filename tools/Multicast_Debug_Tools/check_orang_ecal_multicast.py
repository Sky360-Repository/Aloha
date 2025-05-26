# \copyright    Sky360.org
#
# \brief        Script to verify multicast configuration for Linux platform.
#
# ************************************************************************
#
# Run:#
# chmod +x check_orang_ecal_multicast.py
# ./check_orang_ecal_multicast.py

import os
import subprocess
import re

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def log(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def get_network_interface():
    routes = run("ip route")
    match = re.search(r"default via .* dev (\S+)", routes)
    return match.group(1) if match else None

def check_netplan_route():
    routes = run("ip route show")
    if "239.0.0.0/24 dev lo" in routes and "239.0.0.0/24 dev" in routes:
        log("netplan is set correctly:")
        for line in routes.splitlines():
            if "239.0.0.0/24" in line:
                print(f"    {line}")
        return True
    else:
        warn("Route configuration for 239.0.0.0/24 not found.")
        return False

def get_yaml_for_interface(iface):
    files = run("ls /etc/netplan").splitlines()
    for f in files:
        content = run(f"cat /etc/netplan/{f}")
        if iface in content:
            return f
    return None

def check_firewall_status():
    status = run("sudo ufw status")
    if "inactive" in status.lower():
        log("Firewall is disabled - should have no issues with UDP.")
        return False
    else:
        log("Firewall is enabled.")
        routes = run("ip route show | grep 239.0.0.0")
        if routes:
            log(f"Firewall enabled and route exists:\n    {routes}")
        else:
            warn("Firewall enabled but UDP multicast route missing.")
            iface = get_network_interface()
            if iface:
                print(f"    Add manually with:\nsudo ip route add 239.0.0.0/24 via 192.168.1.254 dev {iface}")
        return True

def check_igmp_membership():
    igmp = run("ip maddr show")
    if "239.0.0.1" in igmp:
        log("IGMP Membership is set.\n       Should see inet 239.0.0.1 under the active network interface:")
        print(f"    {igmp}")
    else:
        warn("IGMP Membership not set for 239.0.0.1.")
        iface = get_network_interface()
        if iface:
            print(f"""    Set it manually with:
    sudo apt install smcroute -y
    sudo smcroute -j {iface} 239.0.0.1""")

def main():
    log("🔍 Starting UDP/IGMP eCAL diagnostics...\n")

    iface = get_network_interface()
    if iface:
        log(f"Detected active network interface: {iface}")
    else:
        warn("Could not detect active network interface.")
        return

    if not check_netplan_route():
        yaml = get_yaml_for_interface(iface)
        if not yaml:
            warn("YAML config file not found for interface. Try running:")
            print("    sudo netplan generate")
        else:
            warn(f"Check route settings in: /etc/netplan/{yaml}")

    check_firewall_status()
    check_igmp_membership()

    log("\n✅ Diagnostics complete.")

if __name__ == "__main__":
    main()
