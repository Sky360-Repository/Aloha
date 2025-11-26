# Airspace Controller

## Modules Used
- **RTL-SDR dongle**
  Used for receiving ADS-B signals from aircraft.

- **GPS module (u-blox)**
  Provides local position, timing reference and list of available satellites.

## Installation on Windows

### RTL-SDR
1. **Driver workaround**
   - By default, Windows installs DVB drivers for the RTL2832 chipset.
   - Replace them with **Zadig drivers**:
     - Download [Zadig](https://zadig.akeo.ie/).
     - Plug in the RTL-SDR dongle.
     - In Zadig, select the device (`Bulk-In, Interface 0`).
     - Install the **WinUSB driver**.
   - This allows `rtl-sdr` tools to access the dongle.

2. **Software**
   - Install [RTL-SDR Blog drivers](https://www.rtl-sdr.com/) or use prebuilt binaries.
   - Test with:
     ```powershell
     rtl_test -t
     ```
   - Run ADS-B decoder:
     ```powershell
     rtl_adsb -V -p 30003
     ```

### GPS
- Connect via USB/COM port (e.g., `COM3`).
- Requires clear sky view (place outside or near a window).
- Use `gpsd` on Linux, or direct serial read in Python on Windows.

## Installation on Orange Pi5+

### Dependencies
```bash
sudo apt update
sudo apt install git build-essential cmake libusb-1.0-0-dev
```

### (Optional but recommended) Udev rules
```bash
sudo cp ../rtl-sdr.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Install rtl-sdr tools
```bash
sudo apt install rtl-sdr
```

### Blacklist kernel DVB drivers
```bash
sudo bash -c "echo 'blacklist dvb_usb_rtl28xxu' > /etc/modprobe.d/no-rtl.conf"
sudo bash -c "echo 'blacklist rtl2832' >> /etc/modprobe.d/no-rtl.conf"
sudo bash -c "echo 'blacklist rtl2830' >> /etc/modprobe.d/no-rtl.conf"
```

Update and reboot:
```bash
sudo depmod -a
sudo reboot
```

Verify:
```bash
lsmod | grep rtl
```
Should produce **no output**.

### Test dongle
```bash
rtl_test -t
```
Expected output:
```
Found 1 device(s):
  0: RTLSDRBlog, Blog V4, SN: 00000001
...
Supported gain values...
```

### Run ADS-B decoder
```bash
rtl_adsb -V -p 30003
```

## Basic Configuration Notes
- **GPS module**
  - Must be placed outside or near a window for clear sky visibility.
  - Provides latitude, longitude, altitude for ADS-B relative positioning.

- **RTL-SDR dongle**
  - Can be indoors, but ideally near a window for better reception.
  - Connect to a suitable antenna (1090 MHz tuned).

## Troubleshooting

### RTL-SDR Issues

- **Device not found**
  - **Windows:** Check that Zadig installed the **WinUSB driver** for `Bulk-In, Interface 0`. If Windows reinstalled DVB drivers, repeat the Zadig step.
  - **Orange Pi/Linux:** Run `lsusb` to confirm the dongle is detected. If not, try another USB port or powered hub.

- **Kernel DVB driver conflict (Linux)**
  - Symptom: `rtl_test -t` fails or device busy.
  - Fix: Ensure you blacklisted `dvb_usb_rtl28xxu`, `rtl2832`, and `rtl2830` in `/etc/modprobe.d/no-rtl.conf`.
    Run `lsmod | grep rtl` — should produce **no output**.

- **Poor reception**
  - ADS-B requires a 1090 MHz antenna. Place the antenna near a window or outside for best results.
  - USB extension cables can help position the dongle away from electrical noise.


### GPS Module Issues

- **No fix / always “invalid”**
  - GPS antennas need clear sky view. Place the module outside or at least near a window.
  - First fix can take up to 30 minutes if the module has no backup battery.

- **COM port not found (Windows)**
  - Check Device Manager → Ports (COM & LPT). Note the COM port number and update your script (`port='COM3'` etc.).
  - If the port disappears, try another USB cable or port.

- **Permission denied (Linux)**
  - Check device permissions
   ```bash
   ls -l /dev/ttyACM0
   ```
   You’ll see something like:
   ```
   crw-rw---- 1 root dialout 166, 0 Dec 25 7:00 /dev/ttyACM0
   ```
   → Only root and members of the `dialout` group can access it.

  - Add your user to the `dialout` group:
    ```bash
    sudo usermod -a -G dialout $USER
    ```
  - Log out and back in (or reboot) for the group membership to take effect.

  - Verify group membership
   ```bash
   groups
   ```
   You should see `dialout` listed.


### Software / Build Issues

- **Deprecation warnings in Python (`_descriptor.FieldDescriptor`)**
  - Cause: Old `.pb2.py` files generated with outdated `protoc`.
  - Fix: Regenerate with `grpcio-tools` matching your installed `protobuf` version:
    ```bash
    python -m grpc_tools.protoc -I ecal_lib/proto_files \
      --python_out=ecal_lib/proto_files \
      ecal_lib/proto_files/adsblog.proto \
      ecal_lib/proto_files/gpslog.proto
    ```

- **`rtl_adsb` not found**
  - Ensure `rtl-sdr` tools are installed (`sudo apt install rtl-sdr`).
  - If you built from source, check that `/usr/local/bin` is in your PATH.


### General Tips

- **Testing RTL-SDR**
  ```bash
  rtl_test -t
  ```
  Confirms the dongle is accessible.

- **Testing GPS**
  - On Linux: `gpsmon /dev/ttyUSB0` or `cgps -s`.
  - On Windows: Use a serial terminal (e.g., PuTTY) to check NMEA sentences.

- **Synchronization**
  - If using GPS for time sync (Chrony), ensure PPS is wired and configured. Without PPS, expect ~20–100 ms jitter.
