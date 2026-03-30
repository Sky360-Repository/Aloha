# GNSS PPS Disciplined Clock Setup
## Waveshare NEO-M8T Timing HAT on Orange Pi 5+

This document explains how to configure an **Orange Pi 5+, 6** with the **Waveshare NEO-M8T GNSS Timing HAT** to create a **PPS-disciplined system clock** using **gpsd** and **chrony** and to propagate time with PPS accurracy to other boards.

We define 2 types of boards:
1. the board with the timing HAT - hostname: S360time
2. the board(s) with no timing HAT - hostname: S360asc, S360ptf, ...

The final system operates as a **Stratum-1 time source** with **sub-microsecond clock accuracy**.

GNSS receivers provide two timing signals:

| Signal | Accuracy | Purpose |
| --- | --- | --- |
| NMEA time | ~10-50 ms | determines correct second |
| PPS pulse | ~5-10 ns | aligns clock precisely |

Chrony combines both signals to discipline the system clock.

---

# 1. Install backup battery on the Waveshare NEO-M8T timing HAT

Insert a **ML1220 rechargeable cell battery** into the battery holder on the **Waveshare NEO-M8T Timing HAT**.

### Why this is important

The GNSS receiver stores information in **Battery-Backed RAM (BBR)**:

- satellite almanac
- receiver configuration
- oscillator calibration
- last known position

Without the battery:

- receiver performs a **cold start every boot**
- configuration may be lost
- slower satellite acquisition

With the battery:

- **warm start within seconds**
- configuration persists
- faster PPS availability

---

# 2. Modify cooling plate for battery clearance for horizontal mounting on S360time

(if you use a perpendicular GPIO extender, you can mount the timing HAT vertically and skip this section)

The **Orange Pi 5+,6** usually includes a **metal cooling plate / heat spreader**.

While the plate leaves an opening for the **GPIO header**, it **does not leave space for the ML1220 battery holder** on the timing HAT.

Therefore a **clearance hole must be created**.

### Required modification

Cut or drill a **16 mm hole** into the cooling plate at the position of the battery holder.

### Procedure

1. Hold the timing HAT above the GPIO header.
2. Mark the battery holder position on the cooling plate.
3. Remove the cooling plate if necessary.
4. Drill a **16 mm hole** at the marked location.
5. Remove metal burrs carefully.
6. Reinstall the cooling plate.

After this modification the timing HAT can sit **flush on the GPIO header**.

---

# 3. Mount the timing HAT on S360time

Mount the **Waveshare NEO-M8T Timing HAT** onto the GPIO header.

### Orientation

The **u-blox chip must face toward the HDMI connectors** of the Orange Pi.

Incorrect orientation may damage the board.

---

# 4. Connect cooling fan on all boards

Connect the fan to the GPIO power pins.

| Pin | Function | Wire |
| --- | --- | --- |
| Pin 4 | +5V | Red |
| Pin 6 | GND | Black |

---

# 5. Configure the GNSS receiver on S360time

Start the configuration script:

```bash
cd tools/time_tools
./setting_timing_hat_PPS.py
```

### Menu commands

Use the following key strokes in the listed order:

| Key | Function |
| --- | --- |
| P | Poll configuration |
| A | Apply configuration |
| S | Save configuration |
| T | Test PPS edges |
| X | Exit |

This configures the **TIMEPULSE output** for **1 Hz PPS aligned to UTC** with **pulseLen = 50ms**.

---

# 6. Install gpsd on all boards

Install GNSS support software.

```bash
sudo apt update
sudo apt install gpsd gpsd-clients
```

`gpsd` reads GNSS data from the serial interface and provides time information to applications.

---

# 7. Configure gpsd on S360time

Edit the gpsd configuration file.

```bash
sudo nano /etc/default/gpsd
```

Configuration:

```
START_DAEMON="true"
USBAUTO="false"
DEVICES="/dev/ttyACM0 /dev/pps0"
GPSD_OPTIONS="-n -G"
```

Restart gpsd:

```bash
sudo systemctl restart gpsd
```

---

# 7. Configure gpsd on all other boards

Edit the gpsd configuration file.

```bash
sudo nano /etc/default/gpsd
```

Configuration:

```
START_DAEMON="true"
USBAUTO="false"
DEVICES=""
GPSD_OPTIONS="-n -G"
```

Restart gpsd:

```bash
sudo systemctl restart gpsd
```

---

# 8. Verify GNSS data on S360time

Run:

```bash
cgps
```

Expected:

- satellites visible
- time updating
- signal strength values

### Important

The GNSS receiver must obtain a **3D fix** before PPS output becomes valid.

Check the **PPS LED on the timing HAT**:

| LED behavior | Meaning |
| --- | --- |
| off | no fix |
| blinking 1 Hz | PPS active |

Without a **3D fix**, PPS will not be generated.

---

# 9. Wiring the boards for PPS signal via GPIO pins

To transport the PPS signal from S360time to other boards like S360asc, we need to couple two physical GPIO pins from board to board.

Assume a jumper wire couple, both ends female, of 20cm length, that is to connect physical GPIO pin 12 on the S360time board with the physical GPIO pin 12 on the S360asc board.

The same goes with physical GPIO pin 14.

If more boards need to be connected, the prior jumper wire couple needs to be doubled, tripled, ... (like you hold a bouque of flowers in your hand)

//We need to find a more practicable way for combining several jumper wire couples for connecting all to the S360time board.

## Verifying the PPS signal

```bash
sudo gpiomon gpiochip3 1 0
```

(note: as soon as you do step #9 and activate the kernel PPS support, gpiomon can't access gpiochip3 1 anymore)

---

# 10. Enable kernel PPS support on all boards

Linux must capture the PPS signal via a GPIO line and expose it as `/dev/pps0`.  

On the Orange Pi 5+ (RK3588), this is done using a **Device Tree Overlay (DTO)** applied by U-Boot.

This requires:

- A **device-tree overlay (`.dtbo`)**
- A **U-Boot script (`overlay.scr`) to apply it at boot**

Create overlay directory:

```bash
sudo mkdir -p /boot/overlay-user
```

Create the device-tree source file:

```bash
sudo nano /boot/overlay-user/neo-m8t-pps.dts
```

Insert:

```
/dts-v1/;
/plugin/;

/ {
    compatible = "rockchip,rk3588";

    fragment@0 {
        target-path = "/";
        __overlay__ {

            gps_pps: pps_gpio {
                compatible = "pps-gpio";
                gpios = <&gpio3 1 0>;
                status = "okay";
            };
        };
    };
};
```

Compile the overlay:

```bash
sudo dtc -O dtb -o /boot/overlay-user/neo-m8t-pps.dtbo -b 0 -@ /boot/overlay-user/neo-m8t-pps.dts
```

Create the overlay script:

```bash
sudo nano /boot/overlay.scr.txt
```

Insert:

```
# Apply PPS overlay via U-Boot

# Point to the active device tree
fdt addr ${fdtcontroladdr}

# Load overlay into RAM
load mmc 0:1 0x43000000 /boot/overlay-user/neo-m8t-pps.dtbo

# Apply overlay
fdt apply 0x43000000
```

Compile the script:

```bash
sudo mkimage -A arm64 -T script -C none -n "PPS Overlay" -d /boot/overlay.scr.txt /boot/overlay.scr
```

Reboot the system:

```bash
sudo reboot
```

---

# 11. Verify PPS device on all boards

After reboot verify PPS is available.

```bash
ll /dev/pps*
```

Expected:

```
/dev/pps0
```

Test PPS capture:

```bash
sudo apt install pps-tools
sudo ppstest /dev/pps0
```

Example output:

```
source 0 - assert 1700000000.000000001
```

This confirms the kernel is receiving PPS pulses.

---

# 12. Install and configure chrony on S360time

Install chrony:

```bash
sudo apt install chrony
```

Edit the configuration:

```bash
sudo nano /etc/chrony/chrony.conf
```

Insert (check for double entries):

```
driftfile /var/lib/chrony/chrony.drift

refclock PPS /dev/pps0 refid PPS lock GPS precision 1e-7
refclock SHM 0 refid GPS precision 1e-1 delay 0.2

ntsdumpdir /var/lib/chrony
logdir /var/log/chrony

maxupdateskew 100.0
rtcsync
makestep 1 3
leapsectz right/UTC
```

Restart chrony:

```bash
sudo systemctl restart chrony
```

---

# 12. Install and configure chrony on all other boards

Install chrony:

```bash
sudo apt install chrony
```

Edit the configuration:

```bash
sudo nano /etc/chrony/chrony.conf
```

Insert (check for double entries):

```
driftfile /var/lib/chrony/chrony.drift

refclock PPS /dev/pps0 refid PPS lock GPS precision 1e-7
server S360time iburst

ntsdumpdir /var/lib/chrony
logdir /var/log/chrony

maxupdateskew 100.0
rtcsync
makestep 1 3
leapsectz right/UTC
```

Restart chrony:

```bash
sudo systemctl restart chrony
```

---

## Explanation of Chrony Configuration (example for S360time)

### driftfile
Stores oscillator drift calibration so chrony learns the system clock characteristics.

### refclock SHM (GPS)

```
refclock SHM 0 refid GPS precision 1e-1 delay 0.2
```

This receives **NMEA time from gpsd** via shared memory.

Purpose:

- identify the correct UTC second
- coarse time source

### refclock PPS

```
refclock PPS /dev/pps0 refid PPS lock GPS precision 1e-7
```

This receives the **precise PPS edge** from the kernel.

Purpose:

- align system clock to nanosecond-level precision

### lock GPS

Ensures the PPS signal is associated with the **correct NMEA second**.

### precision

Expected accuracy of the signal:

| Source | Precision |
| --- | --- |
| GPS NMEA | 100 ms |
| PPS | 100 ns |

### maxupdateskew

Rejects unstable time sources.

### rtcsync

Synchronizes the hardware RTC from the disciplined system clock.

### makestep

Allows large corrections during the first three updates.

### leapsectz

Provides leap second information from the timezone database.

---

# 13. Verify time synchronization

Check chrony sources:

```bash
chronyc sources
```

Expected output on S360time:

```
#* PPS
#- GPS
```

Expected output on all other boards:

```
#* PPS
^- S360time.lan
```

Meaning:

| Symbol | Meaning |
| --- | --- |
| # | reference clock |
| ^ | server clock |
| * | selected time source |
| - | source is available and useful |

Check detailed status:

```bash
chronyc tracking
```

Example:

```
Reference ID    : 50505300 (PPS)
Stratum         : 1
System time     : 0.000000600 seconds fast of NTP time
```

---

# 14. Full validation checklist

### Hardware

- Orange Pi5+ 16GB w hostname: S360time
- optional: Orange Pi5+ 16GB w hostname: S360asc, or S360ptf, ...
- ML1220 battery installed
- timing HAT correctly oriented
- GNSS antenna connected
- PPS LED blinking
- GPIO wires connected

### GNSS on S360time

```
cgps
```

- satellites visible
- **3D fix achieved**

### PPS on all boards

```
ll /dev/pps0
ppstest /dev/pps0
```

- PPS pulses detected

### Chrony on all boards

```
chronyc sources
chronyc tracking
```

Expected:

- PPS selected as source
- system offset < **1 µs**

---

# 15. Final result

The system now operates as a **GNSS disciplined Stratum-1 clock** on all connected boards.

Typical accuracy:

| Component | Accuracy |
| --- | --- |
| GNSS NMEA | ~50 ms |
| PPS pulse | ~5 µs |
| Linux disciplined clock | < 1 µs |

This configuration provides a **high-precision UTC reference clock** suitable for scientific instrumentation, distributed measurement systems, and precision timestamping.
