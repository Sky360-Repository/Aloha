#!/usr/bin/env python3
# coding: utf-8

import h5py
import numpy as np
from multiprocessing import shared_memory
import sys
import os

def load_to_shm(hdf5_path, shm_name_frames="cam_ring_buffer", shm_name_meta="metadata_ring_buffer"):
    # Load HDF5 file
    if not os.path.exists(hdf5_path):
        print(f"❌ File not found: {hdf5_path}")
        sys.exit(1)

    with h5py.File(hdf5_path, 'r') as f:
        frames = f["frames"][:]
        metadata = f["metadata"][:]

    buffer_size, roi_height, roi_width = frames.shape
    print(f"✅ Loaded HDF5: {frames.shape=} {metadata.shape=}")

    # Clean up existing SHM (optional safety step)
    for name in [shm_name_frames, shm_name_meta]:
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
            print(f"⚠️ Removed existing shared memory: {name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ Could not remove existing shared memory {name}: {e}")

    # Create SHM for frames
    frame_bytes = roi_height * roi_width
    shm_frame = shared_memory.SharedMemory(create=True, size=buffer_size * frame_bytes * 2, name=shm_name_frames)
    shm_frame_np = np.ndarray((buffer_size, roi_height, roi_width), dtype=np.uint16, buffer=shm_frame.buf)
    shm_frame_np[:] = frames[:]

    # Create SHM for metadata
    shm_meta = shared_memory.SharedMemory(create=True, size=buffer_size * 3 * 8, name=shm_name_meta)
    shm_meta_np = np.ndarray((buffer_size, 3), dtype=np.float64, buffer=shm_meta.buf)
    shm_meta_np[:] = metadata[:]

    print(f"✅ Successfully loaded data into shared memory.")
    print(f"   SHM names: {shm_name_frames}, {shm_name_meta}")
    print(f"   Frames dtype: {shm_frame_np.dtype}, Metadata dtype: {shm_meta_np.dtype}")

    # Optional: Keep SHM alive until manually killed
    print("🟢 SHM loaded and ready. Press Ctrl+C to exit and release memory.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 Exiting and cleaning up shared memory...")
        shm_frame.close()
        shm_frame.unlink()
        shm_meta.close()
        shm_meta.unlink()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_hdf5_to_shm.py /path/to/qhy_capture_YYYY-MM-DD_*.hdf5")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    load_to_shm(hdf5_path)

