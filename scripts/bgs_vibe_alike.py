#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import time
import keyboard
from collections import deque
from numba import njit, prange
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker

# SETTINGS
LIVE_MODE = True # True is syncing with camera, False starts with SHMs
RING_SIZE = 200
FULL_SHAPE = (3200, 3200)
DOWNSAMPLE = 2
FRAME_SHAPE = (FULL_SHAPE[0] // DOWNSAMPLE, FULL_SHAPE[1] // DOWNSAMPLE)
PIXEL_SIZE = 2
HISTORY_LEN = 16
FG_HISTORY_LEN = 5
MATCH_THRESHOLD = 1600  # [0..32767]
REQUIRED_MATCHES = 2
USE_SMOOTHING = False
SMOOTHING_DECAY = 0.9  # exponential smoothing factor
update_toggle = True
window_title = "Live BGS viewer - press 'q' to quit."
morph_open = 3
morph_close = 3

# Debugging
debug = False
stats = False

# SHM
CAM_SHM_NAME = "cam_ring_buffer"
META_SHM_NAME = "metadata_ring_buffer"
meta_dtype = np.dtype([("timestamp", "f8"), ("exposure", "f8"), ("gain", "f8")])

# Load and downscale the mask to match FRAME_SHAPE
raw_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if raw_mask is None:
    raise RuntimeError("Could not load mask.png")
# Convert to binary and downscale
raw_mask = np.where(raw_mask == 0, 0, 1).astype(np.uint8)
mask_small = cv2.resize(raw_mask, (FRAME_SHAPE[1], FRAME_SHAPE[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

# TODO
# adjust to current exposure and gain changes

# FG smoothing
fg_mask_history = deque(maxlen=FG_HISTORY_LEN)
stable_fg_mask = np.zeros(FRAME_SHAPE, dtype=np.float32)

# Morphing structure
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))

@njit(parallel=True)
def process_frame(curr_frame: np.ndarray,
                  history: np.ndarray,
                  match_threshold: int,
                  required_matches: int,
                  update_: bool,
                  rand_indices: np.ndarray,
                  mask: np.ndarray) -> (np.ndarray, np.ndarray):
    height, width = curr_frame.shape
    fg_mask = np.zeros((height, width), dtype=np.uint8)
    hist_len = history.shape[2]

    for y in prange(height):
        for x in range(width):
            if mask[y, x] == 0:
                continue  # skip masked-out pixels entirely
            px = curr_frame[y, x]
            match_count = 0
            min_val = 0
            max_val = 0
            min_idx = 0
            max_idx = 0

            for i in range(hist_len):
                s = history[y, x, i]
                diff = s - px if s >= px else px - s
                if s > 5 and diff < match_threshold:
                    match_count += 1
                if s < min_val:
                    min_val = s
                    min_idx = i
                elif s > max_val:
                    max_val = s
                    max_idx = i

            if match_count >= required_matches:
                if update_:
                    history[y, x, rand_indices[y, x]] = px
                else:
                    if abs(min_val - px) > abs(max_val - px):
                        history[y, x, min_idx] = px
                    else:
                        history[y, x, max_idx] = px
            else:
                fg_mask[y, x] = 255

    return fg_mask, history

@njit(parallel=True)
def update_stable_mask(stable_fg_mask, vote_mask, decay, mask):
    height, width = stable_fg_mask.shape
    for y in prange(height):
        for x in range(width):
            if mask[y, x] == 0:
                continue  # skip masked pixels
            stable_fg_mask[y, x] = decay * stable_fg_mask[y, x] + (1.0 - decay) * (vote_mask[y, x] / 255.0)
    return stable_fg_mask

def get_newest_index(timestamps: np.ndarray) -> int:
    return int(np.argmax(timestamps))

# Create shared memory for final mask output
mask_frame_shape = (RING_SIZE, *FRAME_SHAPE)
mask_frame_bytes = np.prod(mask_frame_shape) * np.dtype(np.uint8).itemsize
mask_shm = shared_memory.SharedMemory(create=True, size=mask_frame_bytes, name="mask_ring_buffer")

# Connect to SHMs for image and meta data
image_shm = shared_memory.SharedMemory(name=CAM_SHM_NAME)
meta_shm = shared_memory.SharedMemory(name=META_SHM_NAME)
resource_tracker.unregister(image_shm._name, 'shared_memory')
resource_tracker.unregister(meta_shm._name, 'shared_memory')

# Adressing the SHMs
mask_np = np.ndarray(mask_frame_shape, dtype=np.uint8, buffer=mask_shm.buf)
image_np = np.ndarray((RING_SIZE, *FULL_SHAPE), dtype=np.uint16, buffer=image_shm.buf)[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
meta_np = np.ndarray((RING_SIZE,), dtype=meta_dtype, buffer=meta_shm.buf)

# Init history
history = np.zeros((*FRAME_SHAPE, HISTORY_LEN), dtype=np.uint16)
if debug: print("Waiting for sufficient history frames...")
while True:
    if np.count_nonzero(meta_np["timestamp"] > 0) >= HISTORY_LEN:
        break
    time.sleep(0.1)

# Syncing with camera controller
latest_idx = get_newest_index(meta_np["timestamp"])
start_idx = latest_idx if LIVE_MODE else (latest_idx - HISTORY_LEN + 1) % RING_SIZE
ordered_indices = [(start_idx + i) % RING_SIZE for i in range(HISTORY_LEN)]

for i, idx in enumerate(ordered_indices):
    history[:, :, i] = image_np[idx]

if debug: print("History buffer: buffer initialized - start processing loop ...")

prev_latest_idx = start_idx
var_threshold = MATCH_THRESHOLD
num_fg_pixels = 0.0

# TODO: extend/replace with morphology
scales = [
    (20.0, 1, 5),
    (10.0, 1, 5),
    (5.0, 1, 3),
    (1.0, 1, 3),
    (0.1, 1, 1),
    (0.01, 1, 1),
    (0.0, 1, 1),
]

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
                if debug: print("end of offline data.")
                break

        frame = image_np[current_idx]
        prev_latest_idx = current_idx
        start_time = time.perf_counter()

        # Precompute random update indices
        if debug: proc_time = time.perf_counter()
        rand_indices = np.random.randint(HISTORY_LEN, size=FRAME_SHAPE, dtype=np.uint8)
        if debug: print(f"random buffer creation: {(time.perf_counter() - proc_time):.4f}s")
        
        # Dynamic MATCH_THRESHOLD depending on num_fg_pixels
        for limit, m_open, m_close in scales:
            if num_fg_pixels > limit:
                #var_threshold = MATCH_THRESHOLD * factor
                morph_open = m_open
                morph_close = m_close
                break

        # Background subtraction
        if debug: proc_time = time.perf_counter()
        update_toggle = not update_toggle
        fg_mask, history = process_frame(frame, history, var_threshold, REQUIRED_MATCHES, update_toggle, rand_indices, mask_small)
        if debug: print(f"BGS compute: {(time.perf_counter() - proc_time):.4f}s")

        # Morphological noise reduction
        if debug: proc_time = time.perf_counter()
        if morph_open > 1:
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        if morph_close > 1:
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        if debug: print(f"morphing compute: {(time.perf_counter() - proc_time):.4f}s")

        # Temporal smoothing (optional)
        if debug: proc_time = time.perf_counter()
        if USE_SMOOTHING:
            fg_mask_history.append(fg_mask.copy())
            if len(fg_mask_history) == FG_HISTORY_LEN:
                stacked = np.stack(fg_mask_history, axis=0)
                vote_mask = (np.sum(stacked, axis=0) > (255 * FG_HISTORY_LEN // 2)).astype(np.uint8) * 255
            else:
                vote_mask = fg_mask.copy()
            stable_fg_mask = update_stable_mask(fg_mask, vote_mask, SMOOTHING_DECAY, mask_small)
            final_mask = (stable_fg_mask > 0.5).astype(np.uint8) * 255
        else:
            final_mask = fg_mask.copy()

        if debug: print(f"smoothing compute: {(time.perf_counter() - proc_time):.4f}s")
        
        num_fg_pixels = np.count_nonzero(final_mask) / final_mask.size * 100
        print(f"BGS: Index={current_idx} Processing time={(time.perf_counter() - start_time):.4f}s Mask pixels={num_fg_pixels:.4f}% Threshold={int(var_threshold)}")

        # Apply static mask and write final result to SHM
        final_mask[mask_small == 0] = 0
        mask_np[current_idx] = final_mask

        """
        # Debugging
        fps = 1 / (time.perf_counter() - start_time)
        num_fg_pixels = np.count_nonzero(final_mask) / final_mask.size * 100
        if debug: print(f"resulting FPS: {fps:.2f}, mask pixels: {num_fg_pixels:.2f}% -----------------")

        # Statistics
        if stats:
            var_frame = np.var(history, axis=2)
            mean_var = np.mean(var_frame)
            max_var = np.max(var_frame)
            min_var = np.min(var_frame)
            print(f"history variance profile: mean={mean_var:.2f}, max={max_var:.2f}, min={min_var:.2f}")
            var_image = np.clip(np.sqrt(var_frame) / 64.0, 0, 255).astype(np.uint8)  # visual range tweak
            var_heatmap = cv2.applyColorMap(var_image, cv2.COLORMAP_JET)

        # Displaying the BGS results
        frame_display = (frame.astype(np.float32) / 256).astype(np.uint8) # 8bit
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BayerRG2RGB)

        if stats:
            blended = cv2.addWeighted(frame_rgb, 0.3, var_heatmap, 0.7, 0)
        else:
            mask = final_mask != 0
            blended = frame_rgb
            blended[mask] = [0, 0, 255]

        cv2.putText(blended, f"Buffer index: {current_idx}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(blended, f"FPS: {fps:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(blended, f"Foreground pixels: {num_fg_pixels:.2f}%", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(blended, f"Threshold: {MATCH_THRESHOLD} (+/-)", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(blended, f"Smoothing decay: {SMOOTHING_DECAY:.2f} (s/x)", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_title, blended)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if debug: print("Quit requested.")
            break
        elif key == ord('+'):
            MATCH_THRESHOLD += 50
        elif key == ord('-'):
            MATCH_THRESHOLD -= 50
        elif key == ord('s'):
            SMOOTHING_DECAY += 0.01
        elif key == ord('x'):
            SMOOTHING_DECAY -= 0.01
        """


except KeyboardInterrupt:
    print("Interrupted.")

finally:
    image_shm.close()
    meta_shm.close()
    mask_shm.close()
    #cv2.destroyAllWindows()
    print("Cleaned up.")

