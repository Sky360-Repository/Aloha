# Multicast Route Configuration

The example here is multicast using eCAL and connecting Orange PI5+ with a Windows PC. 

Double-check the multicast_group and multicast_port settings in your ecal.ini files:
```
    ecal ini
       multicast_group = 239.0.0.1
       multicast_port = 14000
```

<span style="color:red;font-size:18px;padding:10px;background-color:#f0f0f0"><b>
Important trick to avoid issues: Set fixed IPs on the router!!!!
Multicast didn't work at some point and took me some time to realize that the IP was different :(
</b></span>

### On Windows

```
route -p add 239.0.0.0 mask 255.255.255.0 <your_IP_address>
```
Replace <your_IP_address> with the IP address of your Windows machine.

Verify the route using:
```
route print
```

Get your IP using _ipconfig_

```
Wireless LAN adapter WiFi:
   Connection-specific DNS Suffix  . :
   IPv6 Address. . . . . . . . . . . :
   Temporary IPv6 Address. . . . . . :
   Link-local IPv6 Address . . . . . :
   IPv4 Address. . . . . . . . . . . : XXX.XXX.XXX.XXX
 ```

### On Ubuntu

Follow the instructions from Ecal, they should match what is here.
- https://eclipse-ecal.github.io/ecal/stable/getting_started/cloud.html
- https://eclipse-ecal.github.io/ecal/stable/getting_started/cloud.html#fa-ubuntu-multicast-configuration-on-ubuntu

Netplan uses YAML files for configuration. YAML uses spaces (' ') as indentation and does not work with tabs ('\t').

Add routes section directly under your active interface block, right beneath ***dhcp4: yes*** and ***dhcp6: yes***.

Edit Existing ***/etc/netplan/<File name linked to the network connection>.yaml***

To locate and edit the correct file
```
ls /etc/netplan/
```

This will show you the YAML files present, typically something like 01-netcfg.yaml.

I have: ***90-NM-b6310b0e-716d-4927-a9fa-8880346f9aad.yaml***

```
sudo gedit /etc/netplan/<file-name>.yaml
```

Netplan may give a  warning that the permissions for your YAML file are too permissive. To fix this, run:
```
sudo chmod 600 /etc/netplan/01-network-manager-all.yaml
```

Get the gateway address
```
ip route | grep default
```

My gateway is 192.168.1.254
```
default via 192.168.1.254 dev enP4p65s0 proto dhcp src 192.168.1.186 metric 100
```

My current file ***/etc/netplan/90-NM-b6310b0e-716d-4927-a9fa-8880346f9aad.yaml_*** looks like this:

```
network:
  version: 2
  wifis:
    NM-b6310b0e:
      renderer: NetworkManager
      match:
        name: ""
      dhcp4: true
      dhcp6: true
```

Change it to:
```
network:
  version: 2
  wifis:
    NM-b6310b0e:
      renderer: NetworkManager
      match:
        name: ""
      dhcp4: true
      dhcp6: true
      routes:
        - to: 239.0.0.0/24
          via: 0.0.0.0
          metric: 1
```

Make sure you preserve correct indentation (2 spaces per level).

Now add the Loopback Route, this is a fallback route via loopback that is a new file alongside the existing one:

```
sudo gedit /etc/netplan/50-ecal-multicast-loopback.yaml
```

Then paste:
```
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    lo:
      routes:
        - to: 239.0.0.0/24
          via: 0.0.0.0
          metric: 1000
```

This ensures that if no external interfaces are available, multicast will still resolve via loopback.

<span style="color:red;font-size:18px;padding:10px;background-color:#f0f0f0"><b>
!! We need to do this for all networks. Orange has two Ethernet terminals and I used a USB2WiFi converter.
</b></span>

Apply the changes with:
```
sudo netplan apply
```

Then check the result with:
```
ip route show | grep 239
```

You should see:
```
239.0.0.0/24 dev wlan0 proto static metric 1
239.0.0.0/24 dev lo proto static metric 1000
```

## Firewall Settings:
Ensure that firewalls on both systems allow UDP traffic on the multicast group (e.g., 239.0.0.1) and the relevant ports.

You may need to open ports like 14000 (and others) used by eCAL.

### Windows (Using Windows Defender Firewall):
Open Firewall Settings:
- Go to Control Panel > System and Security > Windows Defender Firewall.
- Click on Advanced Settings.

Create a New Inbound Rule:
- In the left pane, click Inbound Rules.
- In the right pane, click New Rule.

Configure the Rule:
- Select Port and click Next.
- Choose UDP and specify the port number (e.g., 14000), then click Next.
- Select Allow the connection and click Next.
- Choose the network types (Domain, Private, Public) where the rule applies, then click Next.
- Give the rule a name (e.g., "Allow UDP Traffic") and click Finish.

Verify the Rule:
- Ensure the rule is listed and enabled in the Inbound Rules section.

### Ubuntu (Using UFW - Uncomplicated Firewall):

If it says inactive, then your firewall is currently not filtering, and multicast should work already.
```
sudo ufw status
```

If it's active, continue with the steps below.
```
sudo ufw allow 14000/udp comment "Allow udp port"
sudo ufw allow from 239.0.0.1 to any port 14000 proto udp comment "Allow eCAL Multicast"
sudo ufw allow 3389/tcp comment "Allow RDP remote desktop"
sudo ufw enable
```

Note that I use remote desktop to access Orange and I have to allow 3389/tcp.

Check UFW Status and verify the rules are applied:
```
sudo ufw status
```

Always
```
sudo apt -y update && sudo apt -y upgrade && sudo reboot
```

### Verify Multicast Routing on Orange Pi
```
ip route show | grep 239.0.0.0
```

You should see:
```
239.0.0.0/24 via 192.168.1.254 dev <your_interface> proto static metric 600
```

If the route is missing, add the route manually:
```
sudo ip route add 239.0.0.0/24 via 192.168.1.254 dev <your_interface>
```

Replace `<your_interface>` with your actual network device (`enP4p65s0`, etc.).

Check IGMP Membership
```
ip maddr show
```

Look for entries related to `239.0.0.0/24`. If it’s missing, force the group membership:
```
sudo ip maddr add 239.0.0.1 dev <your_interface>
```

If you get Error: Invalid address length 4 - must be 6 bytes
```
sudo apt install smcroute -y
sudo smcroute -j <your_interface> 239.0.0.1
```

<span style="color:red;font-size:18px;padding:10px;background-color:#f0f0f0"><b>
smcroute membership isn't persistent across reboots. 
</b></span>

In my case, Orange is connected through  USB2WiFi converter but I sometimes connect Ethernet. Whenever I switch, I would have to remove the route and add a new one for the other interface. That's why I set the route only when I need.

But if the configuration is permanent, for example it the orange is always connected to the Ethernet port enP4p65s0, then the following configuration will make the membership persistent through reboots.

```
sudo gedit /etc/systemd/system/smcroute-join.service
```

Add the following configuration
```
[Unit]
Description=Join multicast group 239.0.0.1 on <your_interface>
After=network.target

[Service]
ExecStart=/usr/sbin/smcroute -j <your_interface> 239.0.0.1
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

Enable the service
```
sudo systemctl enable smcroute-join.service
sudo systemctl start smcroute-join.service
```

### ecal.ini

Set cloud mode:
- Uses Multicast (239.0.0.1) for registration
- Uses UDP multicast (239.0.0.x) to send data to other hosts
- Uses shared memory to send data to processes on the same host

By default, eCAL is configured in local mode. To switch eCAL to cloud mode, edit your ecal.ini and change the following settings:
- Windows: C:\ProgramData\eCAL\ecal.ini
- Ubuntu: /etc/ecal/ecal.ini

(eCAL monitor has a direct link to ecal.ini)

```
[network]
network_enabled           = true
multicast_ttl             = 10
```

The multicast_ttl setting configures the time to live of the UDP datagrams, i.e. the number of hops a datagram can take before it is discarded. 

I have to set ttl = 10 to send video between Orange and Windows PC.

Set ecal.ini on orange
```
[network]
network_enabled                    = true
multicast_ttl                      = 10
multicast_join_all_if              = true
```

This should match configuration on the PC, except for this parameter that is just for Linux based machines!! ***multicast_join_all_if = true***

### Extra eCAL configurations in eCAL.ini

This can improve the performance but we need to check in real-time it this is not using too much resources.

Just make sure that the eCAL.ini has the same values across devices (except ***multicast_join_all_if*** that is specific to Linux).  

```
[network]
network_enabled                    = true

multicast_ttl                      = 10
; original value was 5242880 = 5Mb; Updated to 8388608 = 8Mb
multicast_sndbuf                   = 5242880
multicast_rcvbuf                   = 5242880

multicast_join_all_if              = false

; Original num_executor = 4; updated to 10
tcp_pubsub_num_executor_reader     = 4
tcp_pubsub_num_executor_writer     = 4

[publisher]

; memfile_minsize = 4096 to 65536
; updated to 64KB minimum allocation
memfile_minsize                    = 4096
; memfile_reserve = 50 to 75
; updated to reserve 75% of memory for buffer
memfile_reserve                    = 50
memfile_ack_timeout                = 0
; memfile_buffer_count = 1 to 5
; Increase buffer count to handle large frames
memfile_buffer_count               = 1
memfile_zero_copy                  = 0
```

## Multicast Debug Tools

Multicast Debug Tools are available in [Aloha/tools/Multicast_Debug_Tools](../tools/Multicast_Debug_Tools)

### Windows

*** Check-EcalMulticast.ps1 *** - This is a PowerShell script to diagnose UDP / IGMP / Multicast Issues on Windows

Run in PowerShell as Administrator:
```
Set-ExecutionPolicy RemoteSigned -Scope Process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Check-EcalMulticast.ps1
```

### Orange
*** check_orang_ecal_multicast.py *** - Orange version of the script to diagnose UDP / IGMP / Multicast Issues

I have an alias on _.bashrc_ that calls this script.

### Orange and windows
*** multicast_diag.py *** - This script is useful to check if two specific devices are communicating. We can define the specific IP and ttl.

Replace 192.168.1.101 with the local IP of the receiver device.
```
python multicast_diag.py recv --iface 192.168.1.101 --ttl 5
```
```
python multicast_diag.py send --iface 192.168.1.101
```

*** pub_sub_handshake.py *** - Simple handshake script using ecal. 

*** pub_sub_handshake_with_sockt.py *** - Handshake script using socket, **it doesn't use ecal**.

*** ttl_logger.py *** - Script listening to UDP MULTICAST_GROUP = '239.0.0.1' and PORT = 14000.

This called ttl_logger because the goal was to test ttl, but this is actually agnostic to ttl.

Use this with ***pub_sub_handshake_with_sockt.py --ttl x***, running in multiple devices with the same ttl value.

If they are not communicating and we see the messages coming through ***ttl_logger.py***, then this means we need to increase ttl value.

# Nothing works :(

Windows is more temperamental and multicast stops working for no reason.

If everything is configured correctly but multicast doesn't work:
- Delete the route
```
route delete 239.0.0.0
```
- Take note of the IP 
```
ipconfig
```
- Set the route again 
```
route -p add 239.0.0.0 mask 255.255.255.0 <Your IP>
```
- also try switching off the firewall
```
  netsh advfirewall set allprofiles state off
```
- Don't forget to switch on the firewall after the tests
```
  netsh advfirewall set allprofiles state on
```
- If it still doesn't work:
```
  Delete the route
  goto Network&Internet->Advanced network settings 
  Scroll down and reset network.
  Reset laptop
```
