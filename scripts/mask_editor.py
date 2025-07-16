#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import time
from multiprocessing import shared_memory, resource_tracker

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"

FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200
VIEWER_WINDOW = 800
PEN_RADIUS = 50  # radius in pixels

debug = False

window_title = "Mask Editor:     s ... save     r ... reset     + ... PEN++     - ... PEN--     q ... quit"

# Attach to shared memories
shm = shared_memory.SharedMemory(name=RING_BUFFER_NAME)
resource_tracker.unregister(shm._name, 'shared_memory')
frame_buffer = np.ndarray((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=shm.buf)

shm2 = shared_memory.SharedMemory(name=METADATA_BUFFER_NAME)
resource_tracker.unregister(shm2._name, 'shared_memory')
metadata_buffer = np.ndarray((FRAME_COUNT, 3), dtype=np.float64, buffer=shm2.buf)

# Create an empty mask: 255 (unmasked), 0 (masked)
mask = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)

# Editor window
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, VIEWER_WINDOW, VIEWER_WINDOW)

drawing = False  # True when mouse is pressed
last_pos = None

def get_newest_index():
    timestamps = metadata_buffer[:, 0]
    newest = np.argmax(timestamps)
    return newest

def draw_circle(event, x, y, flags, param):
    global drawing, mask, last_pos

    scale = FRAME_WIDTH / VIEWER_WINDOW  # assumes square for simplicity
    img_x, img_y = int(x * scale), int(y * scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_pos = (img_x, img_y)
        cv2.circle(mask, last_pos, PEN_RADIUS, 0, -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_pos is not None:
            cv2.line(mask, last_pos, (img_x, img_y), 0, thickness=PEN_RADIUS*2)
            last_pos = (img_x, img_y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_pos = None

cv2.setMouseCallback(window_title, draw_circle)

while True:
    key = cv2.waitKey(10) & 0xFF

    index = (get_newest_index() - 1) % FRAME_COUNT
    img = frame_buffer[index]
    debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

    # Convert 16-bit image to 8-bit for display
    img_8bit = cv2.convertScaleAbs(debayered_img, alpha=255.0/65535.0)

    # Resize and apply mask overlay
    vis_img = cv2.resize(img_8bit, (VIEWER_WINDOW, VIEWER_WINDOW))
    vis_mask = cv2.resize(mask, (VIEWER_WINDOW, VIEWER_WINDOW), interpolation=cv2.INTER_NEAREST)

    # Apply magenta overlay to masked areas
    overlay = vis_img.copy()
    mask_area = vis_mask == 0
    overlay[mask_area, 0] = 255  # Blue
    overlay[mask_area, 1] = 0    # Green
    overlay[mask_area, 2] = 255  # Red

    cv2.putText(overlay, f"PEN: {PEN_RADIUS}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_title, overlay)

    # Save mask
    if key == ord('s'):
        cv2.imwrite("mask.png", mask)
        print("Mask saved as mask.png")

    # Reset mask
    elif key == ord('r'):
        mask[:] = 255
        print("Mask reset.")
    
    elif key == ord('+'):
        PEN_RADIUS += 5

    elif key == ord('-'):
        PEN_RADIUS -= 5
        
    # Quit
    elif key == ord('q'):
        break

    time.sleep(0.01)

# Cleanup
cv2.destroyAllWindows()
shm.close()
shm2.close()
