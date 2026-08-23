#!/usr/bin/env python3
import time
import random
import math

# ============================================================
#  Utility: Dummy CPU + Memory Load
# ============================================================

def cpu_load(duration_ms=50):
    """Burn CPU cycles for a fixed duration."""
    end = time.time() + (duration_ms / 1000.0)
    x = 0.0
    while time.time() < end:
        x += math.sin(random.random())
    return x

def memory_load(size_kb=128):
    """Allocate temporary memory to simulate load."""
    return bytearray(size_kb * 1024)