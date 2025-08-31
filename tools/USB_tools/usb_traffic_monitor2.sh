#!/usr/bin/env python3
# coding: utf-8

import os
import time
import usb.core
import usb.util
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import numpy as np

# =======================
# Config
# =======================
VENDOR_ID = 0x1618  # QHYCCD
PRODUCT_ID = 0xc184  # QHY183c

USBMON_PATH = "/sys/kernel/debug/usb/usbmon/8u"
SAMPLE_INTERVAL = 1.0   # seconds per aggregation
WINDOW_SECONDS = 60     # show last 60 seconds
ROLLING_SECONDS = 10    # rolling average window
STABILIZATION_SECONDS = 10

# =======================
# Globals
# =======================
rates = deque(maxlen=1000)
timestamps = deque(maxlen=1000)
current_rate = 0.0
peak_rate = 0.0
device_name = "Unknown Device"
window_maxlen = int(WINDOW_SECONDS / SAMPLE_INTERVAL)

# =======================
# Find device
# =======================
def find_device():
    global device_name
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is not None:
        try:
            prod = usb.util.get_string(dev, dev.iProduct)
            device_name = f"QHYCCD {prod}"
        except Exception:
            device_name = "QHYCCD (unknown model)"
    else:
        device_name = "Device not found"

# =======================
# Parse usbmon line (only completed transfers)
# =======================
def parse_usbmon_line(line):
    parts = line.split()
    if len(parts) < 6:
        return 0
    if parts[2] != 'C':  # Only completed transfers
        return 0
    try:
        length = int(parts[5])
        return max(length, 0)
    except ValueError:
        return 0

# =======================
# Reader: aggregate bytes per second
# =======================
def reader():
    global current_rate, peak_rate
    if not os.path.exists(USBMON_PATH):
        raise RuntimeError(f"{USBMON_PATH} does not exist. Run with sudo and load usbmon.")

    current_second = int(time.time())
    second_sum = 0

    with open(USBMON_PATH, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.01)
                continue
            size = parse_usbmon_line(line)

            now = int(time.time())
            if now != current_second:
                # Append total bytes of previous second
                rates.append(second_sum)
                timestamps.append(current_second)
                # Update peak rate
                peak_rate = max(peak_rate, second_sum / SAMPLE_INTERVAL / (1024*1024))
                # Reset for new second
                second_sum = 0
                current_second = now

            second_sum += size

# =======================
# Animate
# =======================
def animate(frame):
    global start_time, current_rate, peak_rate
    ax = plt.gca()
    elapsed = time.time() - start_time
    ax.clear()

    # ----------------------------
    # Stabilization period
    # ----------------------------
    if elapsed < STABILIZATION_SECONDS:
        info_text = f"Device: {device_name}\nStabilizing... {elapsed:.1f}/{STABILIZATION_SECONDS}s"
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round", pad=0.5),
            ha="left", va="top"
        )
        return

    # ----------------------------
    # Prepare data
    # ----------------------------
    if not rates:
        return
    rate_list = list(rates)[-window_maxlen:]
    time_list = list(timestamps)[-window_maxlen:]
    hist_rates = [r / (1024*1024) for r in rate_list]  # MB/s
    hist_times = [t - start_time for t in time_list]

    if len(hist_times) < 1:
        return

    # Skip first ROLLING_SECONDS for plotting
    plot_mask = [t >= ROLLING_SECONDS for t in hist_times]
    hist_times_plot = [t for t, m in zip(hist_times, plot_mask) if m]
    hist_rates_plot = [r for r, m in zip(hist_rates, plot_mask) if m]

    if len(hist_times_plot) < 1:
        return

    # Current and peak rates
    current_rate = hist_rates_plot[-1]
    peak_rate = max(hist_rates_plot)

    # Rolling average using 'valid' to avoid edge artifacts
    rolling_len = max(1, int(ROLLING_SECONDS / SAMPLE_INTERVAL))
    if len(hist_rates) >= rolling_len:
        rolling_avg_values_full = np.convolve(hist_rates, np.ones(rolling_len)/rolling_len, mode='valid')
        rolling_times_full = hist_times[rolling_len-1:]
        # Apply same plot mask
        mask = [t >= ROLLING_SECONDS for t in rolling_times_full]
        rolling_times_plot = [t for t, m in zip(rolling_times_full, mask) if m]
        rolling_avg_values_plot = [r for r, m in zip(rolling_avg_values_full, mask) if m]
    else:
        rolling_times_plot = []
        rolling_avg_values_plot = []

    # ----------------------------
    # Plot
    # ----------------------------
    ax.plot(hist_times_plot, hist_rates_plot, label="Current Rate", color="blue")
    if rolling_avg_values_plot:
        ax.plot(rolling_times_plot, rolling_avg_values_plot,
                label=f"{ROLLING_SECONDS}s Rolling Avg", color="green")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Data Rate (MB/s)")
    ax.set_title("Sky360 - USB Data Rate Monitor (ESC to exit)")
    ax.legend(loc="upper right")

    # Info box
    rolling_last = rolling_avg_values_plot[-1] if rolling_avg_values_plot else 0
    info_text = (
        f"Device: {device_name}\n"
        f"Current Rate: {current_rate:.2f} MB/s\n"
        f"Peak Rate: {peak_rate:.2f} MB/s\n"
        f"10s Rolling Avg: {rolling_last:.2f} MB/s"
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round", pad=0.5),
        ha="left", va="top"
    )

# =======================
# Key handler
# =======================
def on_key(event):
    if event.key in ['q', 'escape']:
        plt.close('all')
        os._exit(0)

# =======================
# Main
# =======================
if __name__ == "__main__":
    find_device()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    start_time = time.time()  # record start

    fig = plt.figure(figsize=(8, 8))
    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = animation.FuncAnimation(fig, animate, interval=int(SAMPLE_INTERVAL * 1000))
    plt.show()

