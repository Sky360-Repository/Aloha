# Time from GPS/Glonass U-Blox G-7020

## Set Up gpsd 

Install and configure the GPS daemon:
```
sudo apt update
sudo apt install gpsd gpsd-clients
```

Edit /etc/default/gpsd:
```
DEVICES="/dev/ttyS0"   # Or whatever serial port your GPS is on
GPSD_OPTIONS="-n"
START_DAEMON="true"
```

Start the service:
```
sudo systemctl restart gpsd
```

Test it:
```
cgps   # or gpsmon
```

## Configure chrony to Use GPS

Install chrony:
```
sudo apt install chrony
```

Edit /etc/chrony/chrony.conf to include:
```
# For SHM from gpsd
refclock SHM 0 delay 0.5 refid GPS

# (optional) fallback to NTP servers
server ntp.ubuntu.com iburst
```

Restart chrony:
```
sudo systemctl restart chrony
```

Check sync status:
```
chronyc sources
```

# Use Server Orange (with GPS) as the time authority, and sync the others to it.

## Option 1: Use chrony in a Local NTP Server Setup (Recommended)

On Server Orange:

Ensure GPS time is set up via gpsd + chrony (as discussed previously).

Edit /etc/chrony/chrony.conf to allow client connections:

```
allow 192.168.1.0/24   # Adjust to match your subnet
local stratum 10       # Makes chrony serve time from GPS
```

Restart chrony:
```
    sudo systemctl restart chrony
```

On Client Orange Pis (Camera & Pan&Tilt):

Edit /etc/chrony/chrony.conf:
```
server <ServerOrange-IP> iburst
```

Example:
```
server 192.168.1.10 iburst
``

Comment out or remove other public NTP servers if you want only GPS-derived time.
```
Restart chrony:
sudo systemctl restart chrony
```

Check status:
```
    chronyc sources
```

You should see 192.168.1.10 (your server) as the time source.


# Hybrid Time Sync: GPS First, NTP Fallback

## On Server Orange (with GPS):

Edit `/etc/chrony/chrony.conf` to include both **GPS and NTP sources**:

```
# GPS via gpsd SHM
refclock SHM 0 delay 0.5 refid GPS

# (Optional but recommended) Add fallback NTP servers
server ntp.ubuntu.com iburst
server 0.pool.ntp.org iburst
server 1.pool.ntp.org iburst

# Allow local network clients
allow 192.168.1.0/24

# Let this device serve time even if only one source is reachable
local stratum 10
```

Chrony will **prefer the most accurate source** (GPS), but will **automatically switch to NTP** if GPS is lost.

Check source selection:
```
chronyc sources -v
```

## On Client Orange Pis (Camera & Pan&Tilt):

They will always get time from Server Orange via LAN, as long as:

* The Server Orange is up.
* Chrony is running and responding to NTP requests.

If **Server Orange goes down**, you could configure the clients to fall back to public NTP:

```
server 192.168.1.10 iburst
server 0.pool.ntp.org iburst
```

Chrony will prefer the local server when available.

