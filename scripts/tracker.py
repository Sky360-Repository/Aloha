#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import time
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker
from numba import njit

debug = True

RING_SIZE = 200
MAX_OBJECTS = 100
H, W = 1600, 1600
MAX_LIFETIME = 3
TRACK_ENTRY_LEN = 6 # id, cx, cy, w, h, lifetime
DIST_THRESH = 50.0

MASK_SHM_NAME = "mask_ring_buffer"
META_SHM_NAME = "metadata_ring_buffer"
TRACK_SHM_NAME = "tracking_ring_buffer"

# Connect to mask and metadata SHMs
mask_shm = shared_memory.SharedMemory(name=MASK_SHM_NAME)
resource_tracker.unregister(mask_shm._name, 'shared_memory')
mask_buffer = np.ndarray((RING_SIZE, H, W), dtype=np.uint8, buffer=mask_shm.buf)

meta_shm = shared_memory.SharedMemory(name=META_SHM_NAME)
resource_tracker.unregister(meta_shm._name, 'shared_memory')
metadata_dtype = np.dtype([('timestamp', 'f8'), ('exposure', 'f4'), ('gain', 'f4')])
metadata_buffer = np.ndarray((RING_SIZE,), dtype=metadata_dtype, buffer=meta_shm.buf)

# Create the tracking SHM
track_shm = shared_memory.SharedMemory(name=TRACK_SHM_NAME, create=True, size=RING_SIZE * MAX_OBJECTS * TRACK_ENTRY_LEN * 4)
tracking_buffer = np.ndarray((RING_SIZE, MAX_OBJECTS, TRACK_ENTRY_LEN), dtype=np.int32, buffer=track_shm.buf)
tracking_buffer[:] = -1

# Tracker state
next_id = 1
active_tracks = np.full((MAX_OBJECTS, TRACK_ENTRY_LEN + 1), -1, dtype=np.int32)  # +1 for last_seen

@njit
def find_root(label_set, x):
    while label_set[x] != x:
        label_set[x] = label_set[label_set[x]]
        x = label_set[x]
    return x

@njit
def union(label_set, x, y):
    rx = find_root(label_set, x)
    ry = find_root(label_set, y)
    if rx != ry:
        label_set[ry] = rx

@njit
def connected_components_bbox(binary_image):
    H, W = binary_image.shape
    labels = np.zeros((H, W), dtype=np.int32)
    label_set = np.arange(10000, dtype=np.int32)
    next_label = 1

    for y in range(H):
        for x in range(W):
            if binary_image[y, x] == 0:
                continue

            neighbors = []
            if x > 0 and binary_image[y, x - 1] > 0:
                neighbors.append(labels[y, x - 1])
            if y > 0 and binary_image[y - 1, x] > 0:
                neighbors.append(labels[y - 1, x])

            if len(neighbors) == 0:
                labels[y, x] = next_label
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[y, x] = min_label
                for n in neighbors:
                    union(label_set, min_label, n)

    for y in range(H):
        for x in range(W):
            if labels[y, x] > 0:
                labels[y, x] = find_root(label_set, labels[y, x])

    max_label = np.max(labels)
    bboxes = np.zeros((max_label + 1, 5), dtype=np.int32)  # xmin, ymin, xmax, ymax, area
    bboxes[:, 0] = W
    bboxes[:, 1] = H

    for y in range(H):
        for x in range(W):
            lbl = labels[y, x]
            if lbl > 0:
                bboxes[lbl, 0] = min(bboxes[lbl, 0], x)
                bboxes[lbl, 1] = min(bboxes[lbl, 1], y)
                bboxes[lbl, 2] = max(bboxes[lbl, 2], x)
                bboxes[lbl, 3] = max(bboxes[lbl, 3], y)
                bboxes[lbl, 4] += 1

    valid = bboxes[:, 4] > 0
    out = []
    for i in range(len(bboxes)):
        if valid[i]:
            xmin, ymin, xmax, ymax, area = bboxes[i]
            out.append((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1, area))
    return out

@njit
def compute_distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) ** 0.5

@njit
def find_matches(tracks, num_tracks, cx, cy, index):
    min_dist = 1e9
    best_idx = -1
    for i in range(num_tracks):
        if tracks[i, 0] == -1:
            continue
        last_seen = tracks[i, TRACK_ENTRY_LEN]
        if index_diff(index, last_seen) > MAX_LIFETIME:
            continue
        tx, ty = tracks[i, 1], tracks[i, 2]
        dist = compute_distance(cx, cy, tx, ty)
        if dist < DIST_THRESH and dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

@njit
def index_diff(current, past):
    return (current - past + RING_SIZE) % RING_SIZE

# Tracker update
def update_tracking(index):
    global next_id, active_tracks

    mask = mask_buffer[index]
    blobs = connected_components_bbox(mask)

    for (x, y, w, h, area) in blobs:
        cx, cy = x + w // 2, y + h // 2

        match_idx = find_matches(active_tracks, MAX_OBJECTS, cx, cy, index)

        if match_idx != -1:
            active_tracks[match_idx, 1:5] = (cx, cy, w, h)
            active_tracks[match_idx, 5] += 1
            active_tracks[match_idx, TRACK_ENTRY_LEN] = index
        else:
            slot = np.where(active_tracks[:, 0] == -1)[0]
            if len(slot) > 0:
                slot = slot[0]
                active_tracks[slot] = (next_id, cx, cy, w, h, 1, index)
                next_id = 1 if next_id >= 9999 else next_id + 1 

    # Prune dead tracks
    for i in range(MAX_OBJECTS):
        if active_tracks[i, 0] == -1:
            continue
        if index_diff(index, active_tracks[i, TRACK_ENTRY_LEN]) > MAX_LIFETIME:
            active_tracks[i] = -1

    # Write valid tracks to SHM
    tracking_buffer[index][:] = -1
    out_i = 0
    for i in range(MAX_OBJECTS):
        if active_tracks[i, 0] != -1 and active_tracks[i, 5] >= 1:
            if out_i >= MAX_OBJECTS:
                break
            tracking_buffer[index, out_i] = active_tracks[i, :TRACK_ENTRY_LEN]
            if debug:
                print(f"Frame {index:3d} | ID {active_tracks[i,0]:4d} | Pos ({active_tracks[i,1]},{active_tracks[i,2]}) | Size {active_tracks[i,3]}x{active_tracks[i,4]} | Lifetime {active_tracks[i,5]}")
            out_i += 1

# Main loop
if __name__ == "__main__":
    print("Tracker running...")
    prev_index = -1
    try:
        while True:
            index = (np.argmax(metadata_buffer['timestamp']) - 2) % RING_SIZE # 2 in the pipeline (1 .. BGS, 0 .. camera controller)
            if index != prev_index:
                update_tracking(index)
                prev_index = index
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Tracker stopped.")
    finally:
        mask_shm.close()
        meta_shm.close()
        track_shm.close()
