#!/usr/bin/env python3
# coding: utf-8

# \copyright    Sky360.org
#
# \brief        Configures the NEO-M8T timing hat for PPS output.
#               Frequency: 1Hz
#               pulseLen: 50ms
#               output on GPIO pin 12 (physically)
#
# ************************************************************************

import serial
import time
import struct

try:
    import gpiod
    HAS_GPIOD = True
except ImportError:
    HAS_GPIOD = False
    print("⚠️ gpiod not installed: PPS testing disabled")


# ------------------------
# CONFIGURATION PARAMETERS
# ------------------------
NEW_PULSE_PERIOD = 1_000_000  # µs (1 Hz)
NEW_PULSE_DURATION = 50_000   # µs (50 ms)
NEW_ACTIVE_HIGH = True        # signal is HIGH
NEW_PUSH_PULL = True          # no open drain
NEW_LOCKED_ONLY = True        # PPS only on 3D fix ... False not working due to firmware bug/feature

NEW_ANT_DELAY      = 15       # ns (3m cable) 5ns per meter
TIMEPULSE_INDEX    = 0        # first PPS device

SERIAL_PORT        = "/dev/ttyACM0"
BAUDRATE           = 38400
SER_TIMEOUT        = 2

PPS_GPIOCHIP       = "gpiochip3"
PPS_LINE           = 1


# ------------------------
# UBX checksum
# ------------------------
def ubx_checksum(msg_bytes):
    ck_a = 0
    ck_b = 0
    for b in msg_bytes:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


# ------------------------
# Send UBX frame
# ------------------------
def send_ubx(ser, cls, msg_id, payload):
    msg = bytearray([cls, msg_id,
                     len(payload) & 0xFF,
                     (len(payload) >> 8) & 0xFF]) + payload
    ck_a, ck_b = ubx_checksum(msg)
    frame = bytearray([0xB5, 0x62]) + msg + bytearray([ck_a, ck_b])
    ser.write(frame)
    ser.flush()
    return frame


# ------------------------
# Read UBX frame
# ------------------------
def read_ubx_frame(ser, timeout=2.0):
    ser.timeout = timeout
    start = time.time()

    while True:
        b = ser.read(1)
        if not b:
            if time.time() - start >= timeout:
                return None
            continue

        if b[0] != 0xB5:
            continue

        if ser.read(1)[0] != 0x62:
            continue

        hdr = ser.read(4)
        if len(hdr) < 4:
            return None

        cls, msg_id, lsb, msb = hdr
        length = lsb + (msb << 8)

        payload = ser.read(length)
        ser.read(2)

        return cls, msg_id, payload


# ------------------------
# Parse TP5 payload
# ------------------------
def parse_tp5(payload):
    tp = {}
    tp['tpIdx'] = payload[0]
    tp['version'] = payload[1]

    tp['antCableDelay']  = struct.unpack("<i", payload[4:8])[0]
    tp['rfGroupDelay']   = struct.unpack("<i", payload[8:12])[0]
    tp['freqPeriod']     = struct.unpack("<I", payload[12:16])[0]
    tp['freqPeriodLock'] = struct.unpack("<I", payload[16:20])[0]
    tp['pulseLen']       = struct.unpack("<I", payload[20:24])[0]
    tp['pulseLenLock']   = struct.unpack("<I", payload[24:28])[0]
    tp['flags']          = struct.unpack("<I", payload[28:32])[0]

    flags = tp['flags']

    tp['active']      = bool(flags & (1 << 0))
    tp['lockedOnly']  = bool(flags & (1 << 2))
    tp['isFreq']      = bool(flags & (1 << 3))
    tp['isLength']    = bool(flags & (1 << 4))
    tp['alignToTow']  = bool(flags & (1 << 5))
    tp['activeHigh']  = bool(flags & (1 << 6))

    tp['mode'] = "FREQ" if tp['isFreq'] else "PERIOD"
    tp['pulseLenRatio'] = (
        100 * tp['pulseLen'] / tp['freqPeriod']
        if tp['freqPeriod'] else 0
    )

    return tp


# ------------------------
# Pretty print TP5 parameters
# ------------------------
def print_tp5_block(title, payload):
    tp = parse_tp5(payload)

    print(f"{title}")
    print("--- RAW (HEX)")
    print(" ".join(f"{b:02X}" for b in payload))
    print("--- DECODED")
    print(f"Index:           {tp['tpIdx']}")
    print(f"Version:         {tp['version']}")
    print(f"Antenna Delay:   {tp['antCableDelay']} ns")
    print(f"RF Group Delay:  {tp['rfGroupDelay']/1000:.3f} µs")
    print(f"FreqPeriod:      {tp['freqPeriod']} µs")
    print(f"PulseLen:        {tp['pulseLen']} µs")
    print(f"PulseLenRatio:   {tp['pulseLenRatio']:.2f} %")
    print(f"Mode:            {tp['mode']}")
    print(f"Active High:     {tp['activeHigh']}")
    print(f"Active:          {tp['active']}")
    print(f"Locked Only:     {tp['lockedOnly']}")
    print(f"Align To TOW:    {tp['alignToTow']}")
    print(f"Flags (hex):     0x{tp['flags']:08X}")
    print()


# ------------------------
# Build new TP5 payload (correct flags)
# ------------------------
def build_new_tp5_payload(tp, raw_payload):
    payload = bytearray(raw_payload)

    payload[4:8]   = NEW_ANT_DELAY.to_bytes(4, 'little', signed=True)
    payload[12:16] = NEW_PULSE_PERIOD.to_bytes(4, 'little')
    payload[16:20] = NEW_PULSE_PERIOD.to_bytes(4, 'little')
    payload[20:24] = NEW_PULSE_DURATION.to_bytes(4, 'little')
    payload[24:28] = NEW_PULSE_DURATION.to_bytes(4, 'little')

    flags = tp['flags']

    # Clear bits we control
    flags &= ~(
        (1 << 0) |  # active
        (1 << 2) |  # lockedOnly
        (1 << 3) |  # isFreq
        (1 << 6)    # polarity
    )

    # ACTIVE enable
    flags |= (1 << 0)

    # PERIOD mode → isFreq = 0 (already cleared)
    # POLARITY
    if NEW_ACTIVE_HIGH:
        flags |= (1 << 6)
    # LOCKED ONLY
    if NEW_LOCKED_ONLY:
        flags |= (1 << 2)

    payload[28:32] = flags.to_bytes(4, 'little')

    return payload


# ------------------------
# CHIP RESET (controlled software reboot)
# ------------------------
def chip_reset(ser):
    # UBX-CFG-RST payload: navBbrMask=0, resetMode=1 (controlled software reset)
    payload = struct.pack("<HB", 0x0000, 0x01)
    send_ubx(ser, 0x06, 0x04, payload)
    print("Performed controlled GNSS reset. Waiting 3 seconds...")
    time.sleep(3)
    

# ------------------------
# PPS GPIO test
# ------------------------
def test_pps_gpio():
    if not HAS_GPIOD:
        print("⚠️ gpiod not available")
        return

    print(f"Testing PPS edges on {PPS_GPIOCHIP} line {PPS_LINE}")
    chip = gpiod.Chip(PPS_GPIOCHIP)
    line = chip.get_line(PPS_LINE)
    line.request(consumer='pps_test', type=gpiod.LINE_REQ_DIR_IN)

    last_state = line.get_value()
    print("Press Ctrl+C to exit...\n")

    try:
        while True:
            state = line.get_value()
            if state != last_state:
                print(f"{time.time():.6f} PPS edge: {state}")
                last_state = state
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nExiting PPS test.")


# ------------------------
# MAIN LOOP
# ------------------------
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SER_TIMEOUT)
    except Exception as e:
        print(f"❌ Serial open error: {e}")
        return

    print("✅ Serial OK\n")

    while True:

        ser.reset_input_buffer()
        send_ubx(ser, 0x06, 0x31, bytearray([TIMEPULSE_INDEX]))

        raw_payload = None
        start = time.time()
        while time.time() - start < SER_TIMEOUT:
            resp = read_ubx_frame(ser)
            if not resp:
                continue

            cls, msg_id, payload = resp
            print(f"Received: CLS=0x{cls:02X} ID=0x{msg_id:02X} LEN={len(payload)}")

            if cls == 0x06 and msg_id == 0x31 and len(payload) == 32:
                raw_payload = payload
                break

        if not raw_payload:
            print("❌ No TP5 response\n")
            continue

        tp = parse_tp5(raw_payload)
        new_payload = build_new_tp5_payload(tp, raw_payload)

        # Human-readable print before/after
        print_tp5_block("CURRENT TP5 PARAMETERS:", raw_payload)
        print_tp5_block("NEW TP5 PARAMETERS:", new_payload)

        print("Options: P=Poll  A=Apply  S=Save  T=Test PPS  X=Exit")
        choice = input("> ").strip().upper()

        if choice == "A":
            send_ubx(ser, 0x06, 0x31, new_payload)
            print("Applied.\n")
            time.sleep(0.5)

        elif choice == "S":
            # ------------------------
            # Save to BBR + Flash
            # ------------------------
            clearMask = 0x00000000
            saveMask  = 0x0000FFFF   # save all config blocks
            loadMask  = 0x00000000
            devMask   = 0x03         # BBR + Flash

            payload = struct.pack("<IIIb",
                                  clearMask,
                                  saveMask,
                                  loadMask,
                                  devMask)

            send_ubx(ser, 0x06, 0x09, payload)
            print("Saved to BBR + Flash.\n")
            time.sleep(2)  # wait 2s before reset

            # ------------------------
            # Controlled reset
            # ------------------------
            chip_reset(ser)

            # Flush buffers and wait for module to be ready
            ser.reset_input_buffer()
            time.sleep(2)  # wait extra for GNSS to finish reboot

            # ------------------------
            # Poll TP5 to verify persistence
            # ------------------------
            send_ubx(ser, 0x06, 0x31, bytearray([TIMEPULSE_INDEX]))

            raw_payload = None
            start = time.time()
            while time.time() - start < SER_TIMEOUT:
                resp = read_ubx_frame(ser)
                if not resp:
                    continue
                cls, msg_id, payload = resp
                if cls == 0x06 and msg_id == 0x31 and len(payload) == 32:
                    raw_payload = payload
                    break

            if raw_payload:
                tp = parse_tp5(raw_payload)
                print_tp5_block("TP5 PARAMETERS AFTER SAVE + RESET:", raw_payload)

                # Check if saved values match our desired ones
                if tp['freqPeriod'] == NEW_PULSE_PERIOD and tp['pulseLen'] == NEW_PULSE_DURATION:
                    print("✅ TP5 persistence verified. SAVE was successful!\n")
                else:
                    print("⚠ TP5 persistence mismatch! Check Flash write.\n")
            else:
                print("❌ Could not poll TP5 after reset.\n")

        elif choice == "T":
            test_pps_gpio()

        elif choice == "X":
            break

    ser.close()


if __name__ == "__main__":
    main()
