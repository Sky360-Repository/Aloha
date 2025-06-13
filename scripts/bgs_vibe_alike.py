#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import time
from collections import deque
from numba import njit, prange
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker

# SETTINGS
LIVE_MODE = True  # ← set to False to process SHM sequentially (pre-loaded from hdf5 file)
RING_SIZE = 200
FULL_SHAPE = (3200, 3200)
DOWNSAMPLE = 2
FRAME_SHAPE = (FULL_SHAPE[0] // DOWNSAMPLE, FULL_SHAPE[1] // DOWNSAMPLE)
PIXEL_SIZE = 2
HISTORY_LEN = 30
FG_HISTORY_LEN = 5
MATCH_THRESHOLD = 1600 # [0..32767]
REQUIRED_MATCHES = 2
window_title = "Live BGS viewer - press 'q' to quit."
morph_open = 5
morph_close = 5
USE_RANDOM_UPDATE = True  # set True to use original ViBe random replacement
if USE_RANDOM_UPDATE:
    print("using ViBe random replacement")
else:
    print("using Richards min/max replacement")

# SHM settings
CAM_SHM_NAME = "cam_ring_buffer"
META_SHM_NAME = "metadata_ring_buffer"
meta_dtype = np.dtype([("timestamp", "f8"), ("exposure", "f8"), ("gain", "f8")])

# FG smoothing
fg_mask_history = deque(maxlen=FG_HISTORY_LEN)

@njit(parallel=True)
def process_frame(curr_frame: np.ndarray, history: np.ndarray, match_threshold: int, required_matches: int, use_random_update: bool) -> (np.ndarray, np.ndarray):

    height, width = curr_frame.shape
    fg_mask = np.zeros((height, width), dtype=np.uint8)
    hist_len = history.shape[2]

    for y in prange(height):
        for x in range(width):
            px = curr_frame[y, x]
            samples = history[y, x, :]
            match_count = 0

            min_val = max_val = samples[0]
            min_idx = max_idx = 0

            for i in range(hist_len):
                s = samples[i]
                if abs(int(s) - int(px)) < match_threshold:
                    match_count += 1
                if s < min_val:
                    min_val = s
                    min_idx = i
                elif s > max_val:
                    max_val = s
                    max_idx = i

            if match_count >= required_matches:
                # background
                if use_random_update:
                    # ViBe original: randomly choose a sample to update
                    history[y, x, np.random.randint(hist_len)] = px
                    # changing a second pixel
                    history[y, x, np.random.randint(hist_len)] = px
                    history[y, x, np.random.randint(hist_len)] = px
                else:
                    # Richard logic: update either min or max
                    if abs(int(min_val) - int(px)) > abs(int(max_val) - int(px)):
                        history[y, x, min_idx] = px
                    else:
                        history[y, x, max_idx] = px
            else:
                # foreground
                fg_mask[y, x] = 255

    return fg_mask, history

def get_newest_index(timestamps: np.ndarray) -> int:
    return int(np.argmax(timestamps))

# Attach to shared memory (identical for live and offline mode)
image_shm = shared_memory.SharedMemory(name=CAM_SHM_NAME)
meta_shm = shared_memory.SharedMemory(name=META_SHM_NAME)
resource_tracker.unregister(image_shm._name, 'shared_memory')
resource_tracker.unregister(meta_shm._name, 'shared_memory')

image_np = np.ndarray((RING_SIZE, *FULL_SHAPE), dtype=np.uint16, buffer=image_shm.buf)[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
meta_np = np.ndarray((RING_SIZE,), dtype=meta_dtype, buffer=meta_shm.buf)

# Initialize history
history = np.zeros((*FRAME_SHAPE, HISTORY_LEN), dtype=np.uint16)
print("Waiting for sufficient history frames...")

while True:
    timestamps = meta_np["timestamp"]
    if np.count_nonzero(timestamps > 0) >= HISTORY_LEN:
        break
    print("Waiting for frames...", flush=True)
    time.sleep(0.1)

latest_idx = get_newest_index(meta_np["timestamp"])
if LIVE_MODE:
    start_idx = latest_idx
else:
    start_idx = (latest_idx - HISTORY_LEN + 1) % RING_SIZE

ordered_indices = [(start_idx + i) % RING_SIZE for i in range(HISTORY_LEN)]
for i, idx in enumerate(ordered_indices):
    history[:, :, i] = image_np[idx]

print("History initialized. Starting processing loop...")
prev_latest_idx = start_idx

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, FRAME_SHAPE[1], FRAME_SHAPE[0])

try:
    while True:
        timestamps = meta_np["timestamp"]
        if LIVE_MODE:
            current_idx = get_newest_index(timestamps)
            if current_idx == prev_latest_idx:
                time.sleep(0.001)
                continue
        else:
            current_idx = (prev_latest_idx + 1) % RING_SIZE
            if timestamps[current_idx] == 0:
                print("End of offline data.")
                break

        frame = image_np[current_idx]
        prev_latest_idx = current_idx

        start_time = time.perf_counter()
        fg_mask, history = process_frame(frame, history, MATCH_THRESHOLD, REQUIRED_MATCHES, USE_RANDOM_UPDATE)

        if morph_open > 1:
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((morph_open, morph_open), np.uint8))
        if morph_close > 1:
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((morph_close, morph_close), np.uint8))

        fg_mask_history.append(fg_mask.copy())
        if len(fg_mask_history) == FG_HISTORY_LEN:
            stacked = np.stack(fg_mask_history, axis=0)
            stable_fg_mask = (np.sum(stacked, axis=0) > (255 * FG_HISTORY_LEN // 2)).astype(np.uint8) * 255
        else:
            stable_fg_mask = fg_mask

        fps = 1 / (time.perf_counter() - start_time)
        num_fg_pixels = np.count_nonzero(stable_fg_mask) / stable_fg_mask.size * 100
        print(f"Frame {current_idx}: {fps:.2f} FPS, foreground pixels: {num_fg_pixels:.2f}%")

        # Convert 16-bit frame to 8-bit for display
        frame_display = (frame.astype(np.float32) / 256).astype(np.uint8)

        # Convert to 3-channel RGB
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BayerRG2RGB)

        # Colorize the foreground mask (e.g., red)
        overlay_color = np.zeros_like(frame_rgb)
        overlay_color[:, :, 2] = stable_fg_mask  # Red channel only

        # Blend the overlay with the original frame
        alpha = 0.8  # Transparency factor: 0 = transparent, 1 = opaque
        blended = cv2.addWeighted(frame_rgb, 1.0, overlay_color, alpha, 0)

        # Display the result
        cv2.imshow(window_title, blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested.")
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    image_shm.close()
    meta_shm.close()
    cv2.destroyAllWindows()
    print("Cleaned up.")

