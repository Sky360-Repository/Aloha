# \copyright    Sky360.org
#
# \brief        eCAL RTOP publisher.
#
# ************************************************************************

import subprocess
import time

from ecal_lib.src.proto_sender import ProtoSender

# Wrapper message
CHANNEL_NAME = "system_metrics"
WRAPPER_MESSAGE_NAME = "SystemMetrics"
PROTO_MODULE = "ecal_lib.proto_files.SystemMetrics_pb2"


# Create sender
metrics_sender = ProtoSender(
    channel_name=CHANNEL_NAME,
    wrapper_message_name=WRAPPER_MESSAGE_NAME,
    proto_module=PROTO_MODULE
)


# Helper function to run shell commands
def run_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except Exception:
        return ""


# CPU usage calculation helper
def get_cpu_times():
    cpu_stats = {}
    with open("/proc/stat", "r") as f:
        for line in f:
            if line.startswith("cpu") and len(line.split()) > 1:
                parts = line.split()
                cpu_id = parts[0]
                times = list(map(int, parts[1:]))
                cpu_stats[cpu_id] = times
    return cpu_stats


def calculate_cpu_usage(prev, curr):
    usage = {}
    for cpu_id in curr:
        if cpu_id in prev:
            prev_total = sum(prev[cpu_id])
            curr_total = sum(curr[cpu_id])
            total_diff = curr_total - prev_total

            idle_diff = curr[cpu_id][3] - prev[cpu_id][3]  # idle is 4th field

            if total_diff != 0:
                cpu_usage = 100.0 * (total_diff - idle_diff) / total_diff
            else:
                cpu_usage = 0.0

            usage[cpu_id] = round(cpu_usage, 1)
    return usage


# CPU frequency
def get_cpu_freq(cpu_index):
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_cur_freq"
        with open(path, "r") as f:
            return round(int(f.read().strip()) / 1e6, 2)  # GHz
    except Exception:
        return 0.0


# Memory usage
def get_mem_info():
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, val = line.split(":")[0], line.split(":")[1].strip().split()[0]
            meminfo[key] = int(val)

    mem_total = meminfo.get("MemTotal", 0)
    mem_free = meminfo.get("MemAvailable", 0)
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)

    mem_used = mem_total - mem_free
    swap_used = swap_total - swap_free

    mem_percent = round(mem_used / mem_total * 100, 1) if mem_total else 0
    swap_percent = round(swap_used / swap_total * 100, 1) if swap_total else 0

    return (
        f"{mem_percent}% / {round(mem_used / 1024 / 1024, 2)} GB",
        f"{swap_percent}% / {round(swap_used / 1024 / 1024, 2)} GB"
    )


# Temperature
def get_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_milli = int(f.read().strip())
            return f"{temp_milli / 1000:.1f}°C"
    except Exception:
        return "N/A"


# GPU load
def get_gpu_info():
    try:
        load = int(run_command("cat /sys/class/devfreq/fb000000.gpu/load"))
        freq = int(run_command("cat /sys/class/devfreq/fb000000.gpu/cur_freq"))
        return f"{load}% / {round(freq / 1e6, 2)} GHz"
    except Exception:
        return "N/A"


# Dummy NPU and RGA values (replace with real ones if available)
def get_dummy_npu_rga():
    return [
        "10% / 1.0 GHz", "20% / 1.0 GHz", "15% / 1.0 GHz"
    ], [
        "5% / 0.8 GHz", "7% / 0.9 GHz"
    ]


def main():
    prev_cpu_times = get_cpu_times()
    time.sleep(1)

    while True:
        curr_cpu_times = get_cpu_times()
        cpu_usage = calculate_cpu_usage(prev_cpu_times, curr_cpu_times)

        # IMPORTANT:
        # ProtoSender.get_message_type() returns the payload type (wrapper.data),
        # so here this is pb.sky.Metrics, not pb.sky.SystemMetrics
        metrics = metrics_sender.get_message_type()

        # CPU (first 8 cores)
        for i in range(8):
            usage = cpu_usage.get(f"cpu{i}", 0.0)
            freq = get_cpu_freq(i)
            metrics.cpu.append(f"CPU{i + 1} {usage}% / {freq} GHz")

        # NPU and RGA
        npu_vals, rga_vals = get_dummy_npu_rga()
        metrics.npu.extend([f"NPU{i + 1} {val}" for i, val in enumerate(npu_vals)])
        metrics.rga.extend([f"RGA{i + 1} {val}" for i, val in enumerate(rga_vals)])

        # GPU
        metrics.gpu = get_gpu_info()

        # Memory and Swap
        mem_str, swap_str = get_mem_info()
        metrics.mem = mem_str
        metrics.swap = swap_str

        # Temperature
        metrics.temp = get_temperature()

        # Send payload only — ProtoSender wraps it into SystemMetrics and adds header automatically
        metrics_sender.send(metrics)

        prev_cpu_times = curr_cpu_times
        time.sleep(1)


if __name__ == "__main__":
    main()
