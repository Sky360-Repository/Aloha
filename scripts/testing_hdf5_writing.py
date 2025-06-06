#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import h5py

frames = np.random.randint(0, 65535, (200, 3200, 3200), dtype=np.uint16)
metadata = np.random.rand(200, 3)

with h5py.File("/tmp/test_valid.hdf5", "w") as f:
    f.create_dataset("frames", data=frames, compression="gzip")
    f.create_dataset("metadata", data=metadata, compression="gzip")

print("✅ Successfully wrote test_valid.hdf5")

