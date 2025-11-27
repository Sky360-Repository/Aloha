
# Orange Pi 5+ Ubuntu – ZRAM Swap

Ubuntu typically enables swap via a disk-backed swapfile (on SD or NVMe).

Using SD cards for swap is not recommended due to its limited write endurance.

NVMe drives handle swap better but still incur wear.

A better option is compressed RAM-based swap (`/dev/zram0`), which:
- Reduces wear on SD/NVMe storage
- Provides faster swap performance
- Improves responsiveness under memory pressure

We’ll configure ZRAM to use ~50% of system RAM (≈8 GB on a 16 GB system).

## **Steps**

### 1. Verify kernel supports ZRAM

```bash
sudo modprobe zram
ls /dev/zram*
```

* Must see `/dev/zram0`
* If not, kernel does not support ZRAM → need a different kernel/image.


### 2. Check existing swap

```bash
swapon --show
```

* If swapfile exists (e.g., `/swapfile`), disable it:

```bash
sudo swapoff /swapfile
sudo rm /swapfile
```

### 3. Adjust zramswap configuration

Edit `/etc/default/zramswap`:

```ini
# Compression algorithm (must be supported by kernel)
# Check supported algorithms: cat /sys/block/zram0/comp_algorithm
ALGO=lzo        # Use lzo if lz4 fails

# Use percentage of total RAM
PERCENT=50

# Swap priority
PRIORITY=100
```

### 4. Enable and start zramswap service

```bash
sudo systemctl daemon-reload
sudo systemctl enable zramswap --now
```

Check status:

```bash
swapon --show
systemctl status zramswap.service
```

Expected output:

```
NAME       TYPE      SIZE USED PRIO
/dev/zram0 partition 7.8G   0B  100
```

### 5. Optional tuning

* **Reduce swapping aggressiveness**:

```bash
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

* **View ZRAM stats**:

```bash
zramctl
```

* **Reset ZRAM swap if needed**:

```bash
sudo swapoff /dev/zram0
echo 1 | sudo tee /sys/block/zram0/reset
sudo systemctl restart zramswap
```

## 🚫 Disabling Swap Entirely

Pros:
    - No RAM overhead reserved for swap.
    - No risk of SD/NVMe wear.

Cons:
    - When memory runs out, the kernel has no fallback → processes get killed (OOM killer).
    - Applications that expect swap (e.g., browsers, compilers, ML workloads) may crash under heavy load.
    - System responsiveness drops sharply once RAM is exhausted.
    
ZRAM is a safety valve:
- Without swap, once RAM is full, the system reboots.
- With ZRAM, we trade a slice of RAM for compressed “overflow space”, which can store more than its raw size thanks to compression. 8 GB of ZRAM can hold ~12–16 GB worth of memory pages depending on workload.






