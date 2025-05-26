#!/usr/bin/env python3
# coding: utf-8

import time
import numpy as np
import cv2
from multiprocessing import shared_memory

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"
FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200

# Attach to shared memory
shm = shared_memory.SharedMemory(name=RING_BUFFER_NAME)
frame_buffer = np.ndarray((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=shm.buf)

shm2 = shared_memory.SharedMemory(name=METADATA_BUFFER_NAME)
metadata_buffer = np.ndarray((FRAME_COUNT, 3), dtype=np.float64, buffer=shm2.buf)

# Viewer state
last_timestamp = 0.0
index = 0
frame_counter = 0
fps_timer_start = time.time()
window_title = "Live Viewer - Press 'q' to quit"

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, 800, 800)

print("Viewer: Running. Press 'q' to quit.")

while True:
    ts = metadata_buffer[index][0]

    if ts != last_timestamp:
        last_timestamp = ts
        frame_counter += 1

        img = frame_buffer[index]
        debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        scaled_img = debayered_img[::4, ::4]

        # Format timestamp
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}"
        cv2.putText(scaled_img, f"Timestamp: {timestamp_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate and show FPS every second
        now = time.time()
        elapsed = now - fps_timer_start
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            cv2.setWindowTitle(window_title, f"Live Viewer - {fps:.1f} FPS - Press 'q' to quit")
            frame_counter = 0
            fps_timer_start = now

        cv2.imshow(window_title, scaled_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    index = (index + 1) % FRAME_COUNT
    time.sleep(0.01)

# Cleanup
cv2.destroyAllWindows()
shm.close()
shm2.close()

