#!/usr/bin/env python3
# coding: utf-8

import time
import numpy as np
import cv2
import keyboard
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"
FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200

# Attach to shared memory
shm = shared_memory.SharedMemory(name=RING_BUFFER_NAME)
resource_tracker.unregister(shm._name, 'shared_memory')
frame_buffer = np.ndarray((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=shm.buf)

shm2 = shared_memory.SharedMemory(name=METADATA_BUFFER_NAME)
resource_tracker.unregister(shm2._name, 'shared_memory')
metadata_buffer = np.ndarray((FRAME_COUNT, 3), dtype=np.float64, buffer=shm2.buf)

# Viewer state
last_timestamp = 0.0
current_index = 0
frame_counter = 0
fps_timer_start = time.time()
window_title = "Live Viewer - Press 'q' to quit"
paused = False

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, 800, 800)

def get_newest_index():
    timestamps = metadata_buffer[:, 0]
    newest = np.argmax(timestamps)
    return newest

def show_frame(index):
    img = frame_buffer[index]
    debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    scaled_img = debayered_img[::4, ::4]
    scaled_img_8bit = cv2.convertScaleAbs(scaled_img, alpha=(255.0/65535.0))

    cv2.putText(scaled_img_8bit, f"Index: {index}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    #cv2.putText(scaled_img_8bit, f"Timestamp: {metadata_buffer[index][0]}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(scaled_img_8bit, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(metadata_buffer[index][0]))}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(scaled_img_8bit, f"Exposure: {metadata_buffer[index][1]:.2f} ms", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(scaled_img_8bit, f"Gain: {metadata_buffer[index][2]:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_title, scaled_img_8bit)

while True:
    key = cv2.waitKey(10) & 0xFF

    if not paused:
        newest_index = get_newest_index()
        if metadata_buffer[newest_index][0] != last_timestamp:
            current_index = newest_index
            last_timestamp = metadata_buffer[current_index][0]
            frame_counter += 1
            show_frame(current_index)

            now = time.time()
            elapsed = now - fps_timer_start
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                cv2.setWindowTitle(window_title, f"Live Viewer - {fps:.1f} FPS - Press 'q' to quit")
                frame_counter = 0
                fps_timer_start = now
    else:
        if key == ord('d') or key == 83:  # Right arrow or 'd'
            current_index = (current_index + 1) % FRAME_COUNT
            show_frame(current_index)
        elif key == ord('a') or key == 81:  # Left arrow or 'a'
            current_index = (current_index - 1) % FRAME_COUNT
            show_frame(current_index)

    # Handle toggle and quit
    if key == ord(' '):  # spacebar
        paused = not paused
        print("⏯️ Paused" if paused else "▶️ Resumed")
    elif key == ord('q'):
        break

    key = cv2.waitKey(1) & 0xFF
    time.sleep(0.01)

# Cleanup
cv2.destroyAllWindows()
shm.close()
shm2.close()

