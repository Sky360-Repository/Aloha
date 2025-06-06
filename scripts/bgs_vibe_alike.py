#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import time
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker
from numba import njit, prange
from collections import deque

# Constants
CAM_SHM_NAME = "cam_ring_buffer"
META_SHM_NAME = "metadata_ring_buffer"
RING_SIZE = 200
FULL_SHAPE = (3200, 3200)
DOWNSAMPLE = 4 # 1 .. 7-9FPS, 2 .. 27-31FPS, 4 .. 82-100FPS
FRAME_SHAPE = (FULL_SHAPE[0] // DOWNSAMPLE, FULL_SHAPE[1] // DOWNSAMPLE)  # 800x800px
PIXEL_SIZE = 2  # uint16
HISTORY_LEN = 50
FG_HISTORY_LEN = 5
MATCH_THRESHOLD = 6400 * 3 # lower = stricter match → more foreground, higher = looser match → less foreground
REQUIRED_MATCHES = 2 # higher = harder to be background → more foreground, lower = easier to be background → less foreground
window_title = "BGS static test viewer - press 'q' to quit."
morph_open = 3 # [1,3,5,7, ...]
morph_close = 3 # [1,3,5,7, ...]
random_max = 8

# Derivates
fg_mask_history = deque(maxlen=FG_HISTORY_LEN)

# Processing frame-wise in a JIT binary 
@njit(parallel=True)
def process_frame(curr_frame: np.ndarray, history: np.ndarray,
                  match_threshold: int, required_matches: int) -> (np.ndarray, np.ndarray):
    height, width = curr_frame.shape
    fg_mask = np.zeros((height, width), dtype=np.uint8)
    hist_len = history.shape[2]

    for y in prange(height):
        for x in range(width):
            px = curr_frame[y, x]
            samples = history[y, x, :]
            match_count = 0

            min_val = samples[0]
            max_val = samples[0]
            for i in range(hist_len):
                s = samples[i]
                if abs(int(s) - int(px)) < match_threshold:
                    match_count += 1
                if s < min_val:
                    min_val = s
                if s > max_val:
                    max_val = s

            if match_count >= required_matches:
                # pixel is background
                if np.random.randint(random_max) == 1:
                    if min_val <= px <= max_val:
                        # Replace a random index not equal to min or max
                        min_idx = -1
                        max_idx = -1
                        for i in range(hist_len):
                            if samples[i] == min_val and min_idx == -1:
                                min_idx = i
                            elif samples[i] == max_val and max_idx == -1:
                                max_idx = i
                        # Choose any index but min_idx and max_idx
                        while True:
                            ri = np.random.randint(hist_len)
                            if ri != min_idx and ri != max_idx:
                                break
                        history[y, x, ri] = px
                    else:
                        # Replace the one farther from px (either min or max)
                        if abs(int(min_val) - int(px)) > abs(int(max_val) - int(px)):
                            # Replace min
                            for i in range(hist_len):
                                if samples[i] == min_val:
                                    history[y, x, i] = px
                                    break
                        else:
                            # Replace max
                            for i in range(hist_len):
                                if samples[i] == max_val:
                                    history[y, x, i] = px
                                    break
            else:
                # pixel is foreground
                fg_mask[y, x] = 255

    return fg_mask, history

# Connect to SHMs
image_bytes = np.prod(FULL_SHAPE) * PIXEL_SIZE * RING_SIZE
meta_dtype = np.dtype([("timestamp", "d"), ("exposure", "f"), ("gain", "f")])
meta_bytes = RING_SIZE * meta_dtype.itemsize

image_shm = shared_memory.SharedMemory(name=CAM_SHM_NAME)
meta_shm = shared_memory.SharedMemory(name=META_SHM_NAME)

resource_tracker.unregister(image_shm._name, 'shared_memory')
resource_tracker.unregister(meta_shm._name, 'shared_memory')

# Downsampled image view
image_np = np.ndarray((RING_SIZE, *FULL_SHAPE), dtype=np.uint16, buffer=image_shm.buf)[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
meta_np = np.ndarray((RING_SIZE,), dtype=meta_dtype, buffer=meta_shm.buf)

# Sort by timestamp
#sorted_indices = np.argsort(meta_np["timestamp"])
start_idx = np.argmin(meta_np["timestamp"])
ordered_indices = [(start_idx + i) % RING_SIZE for i in range(RING_SIZE)]

# Init history
history = np.zeros((*FRAME_SHAPE, HISTORY_LEN), dtype=np.uint16)

try:
    print(f"Initializing history with first {HISTORY_LEN} frames...")
    for i in range(HISTORY_LEN):
        frame = image_np[ordered_indices[i]]
        history[:, :, i] = frame
    print("History initialized.")

    print("Running BGS on remaining frames...")
    for rel_idx in range(HISTORY_LEN, RING_SIZE):
        idx = ordered_indices[rel_idx]
        frame = image_np[idx]

        start_time = time.perf_counter()
        fg_mask, history = process_frame(frame, history, MATCH_THRESHOLD, REQUIRED_MATCHES)
        
        # Apply morphological opening to remove small noise patches
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((morph_open, morph_open), np.uint8))
        
        # Apply morphological closing to fill small holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((morph_close, morph_close), np.uint8))
        
        # Begin temporal smoothing
        fg_mask_history.append(fg_mask.copy())
        if len(fg_mask_history) == FG_HISTORY_LEN:
            stacked = np.stack(fg_mask_history, axis=0)
            stable_fg_mask = (np.sum(stacked, axis=0) > (255 * FG_HISTORY_LEN // 2)).astype(np.uint8) * 255
        else:
            stable_fg_mask = fg_mask
        
        fps = 1 / (time.perf_counter() - start_time)
        num_fg_pixels = np.count_nonzero(stable_fg_mask) / stable_fg_mask.size * 100 # [%]
        print(f"Frame {rel_idx}/{RING_SIZE}: {fps:.2f} FPS, foreground pixels: {num_fg_pixels:.2f}%")
        
        cv2.imshow(window_title, stable_fg_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested.")
            break
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    image_shm.close()
    meta_shm.close()
    cv2.destroyAllWindows()
    print("Cleaned up.")

