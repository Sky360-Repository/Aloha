# GNSS PPS Disciplined Clock Setup
## Waveshare NEO-M8T Timing HAT on Orange Pi 5+

This document explains how to configure an **Orange Pi 5+, 6** with the **Waveshare NEO-M8T GNSS Timing HAT** to create a **PPS-disciplined system clock** using **gpsd** and **chrony**.

The final system operates as a **Stratum-1 time source** with **sub-microsecond clock accuracy**.

GNSS receivers provide two timing signals:

| Signal | Accuracy | Purpose
| NMEA time | ~10-50 ms | determines correct second
| PPS pulse | ~5-10 ns | aligns clock precisely

Chrony combines both signals to discipline the system clock.

---

# 1 Install Backup Battery

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

# 2 Modify Cooling Plate for Battery Clearance

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

# 3 Mount the Timing HAT

Mount the **Waveshare NEO-M8T Timing HAT** onto the GPIO header.

### Orientation

The **u-blox chip must face toward the HDMI connectors** of the Orange Pi.

Incorrect orientation may damage the board.

---

# 4 Connect Cooling Fan

Connect the fan to the GPIO power pins.

| Pin | Function | Wire |
| Pin 4 | +5V | Red |
| Pin 6 | GND | Black |

---

# 5 Configure the GNSS Receiver

Start the configuration script:

```bash
cd tools/time_tools
./setting_timing_hat_PPS.py
```

### Menu commands

Use the following key strokes in the listed order:

| Key | Function |
| P | Poll configuration |
| A | Apply configuration |
| S | Save configuration |
| T | Test PPS edges |
| X | Exit |

This configures the **TIMEPULSE output** for **1 Hz PPS aligned to UTC** with **pulseLen = 50ms**.

---

# 6 Install gpsd

Install GNSS support software.

```bash
sudo apt update
sudo apt install gpsd gpsd-clients
```

`gpsd` reads GNSS data from the serial interface and provides time information to applications.

---

# 7 Configure gpsd

Edit the gpsd configuration file.

```bash
sudo nano /etc/default/gpsd
```

Example configuration:

```
START_DAEMON="true"
USBAUTO="false"
DEVICES="/dev/ttyACM0"
GPSD_OPTIONS="-n"
```

Restart gpsd:

```bash
sudo systemctl restart gpsd
```

---

# 8 Verify GNSS Data

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
| off | no fix |
| blinking 1 Hz | PPS active |

Without a **3D fix**, PPS will not be generated.

---

# 9 Enable Kernel PPS Support

Linux must capture the PPS signal using the **pps_gpio kernel driver**.

This requires a **device-tree overlay**.

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

            pps_gpio {
                compatible = "pps-gpio";
                gpios = <&gpio3 1 0>;
                assert-falling-edge;
                status = "okay";
            };

        };
    };
};
```

Compile the overlay:

```bash
dtc -O dtb -o /boot/overlay-user/neo-m8t-pps.dtbo -b 0 -@ /boot/overlay-user/neo-m8t-pps.dts
```

Enable the overlay:

```bash
sudo nano /boot/armbianEnv.txt
```

Add:

```
user_overlays=neo-m8t-pps
```

Reboot the system:

```bash
sudo reboot
```

---

# 10 Verify PPS Device

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
ppstest /dev/pps0
```

Example output:

```
source 0 - assert 1700000000.000000001
```

This confirms the kernel is receiving PPS pulses.

---

# 11 Install and Configure Chrony

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

## Explanation of Chrony Configuration

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

# 12 Verify Time Synchronization

Check chrony sources:

```bash
chronyc sources
```

Expected output:

```
#* PPS
#- GPS
```

Meaning:

| Symbol | Meaning |
| # | reference clock |
| * | selected time source |
| - | source is available and useful |

Check detailed status:

```bash
chronyc tracking
```

Example:

```
Reference ID    : PPS
Stratum         : 1
System time     : 0.000000600 seconds fast
```

---

# 13 Full Validation Checklist

### Hardware

- ML1220 battery installed
- timing HAT correctly oriented
- GNSS antenna connected
- PPS LED blinking

### GNSS

```
cgps
```

- satellites visible
- **3D fix achieved**

### PPS

```
ll /dev/pps0
ppstest /dev/pps0
```

- PPS pulses detected

### Chrony

```
chronyc sources
chronyc tracking
```

Expected:

- PPS selected as source
- system offset < **1 µs**

---

# 14 Final Result

The system now operates as a **GNSS disciplined Stratum-1 clock**.

Typical accuracy:

| Component | Accuracy |
| GNSS NMEA | ~50 ms |
| PPS pulse | ~5 µs |
| Linux disciplined clock | < 1 µs |

This configuration provides a **high-precision UTC reference clock** suitable for scientific instrumentation, distributed measurement systems, and precision timestamping.
