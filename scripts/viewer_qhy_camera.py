#!/usr/bin/env python3
# coding: utf-8

import time
from datetime import datetime, timezone
import numpy as np
import cv2
import keyboard
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker
import json
import os
from numba import njit
from math import radians, degrees, sin, cos, atan2, sqrt, asin

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"
MASK_BUFFER_NAME = "mask_ring_buffer"
FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200
BGS_DOWNSAMPLE = 2
VIEWER_WINDOW = 800
AZIMUTH_OFFSET_DEG = 90
FOV = 160.0

AIRCRAFT_JSON_PATH = os.path.expanduser("~/dump1090-master/dump1090/public_html/data/aircraft.json")
OBSERVER_LAT = 47.647535944954846
OBSERVER_LON = 14.841571479015096
OBSERVER_ALT = 850

debug = False
aircraft_screen_coords = []

# Viewer states
last_timestamp = 0.0
current_index = 0
frame_counter = 0
fps_timer_start = time.time()
window_title = "i ... info     a ... ADSB     b ... BGS     q ... quit"
paused = False
bgs_view = False
info_view = False
adsb_view = False

center = (VIEWER_WINDOW // 2, VIEWER_WINDOW // 2)
outer_radius = VIEWER_WINDOW // 2
inner_radius = int((90.0 / 160.0) * outer_radius)

#plane_icon = cv2.imread("plane.png", cv2.IMREAD_UNCHANGED)
PLANE_POLY = np.array([
    [0.0, -30.0],    # nose tip
    [6.0, -23.0],
    [7.0, -10.0],
    [37.0,  10.0],
    [37.0,  19.0],
    [6.0,  11.0],
    [3.0,   34.0],
    [13.0,   40.0],
    [13.0,   47.0],
    [0.0,   42.0],   # center tail end
    [-13.0,   47.0],
    [-13.0,   40.0],
    [-3.0,  34.0],
    [-6.0, 11.0],
    [-37.0, 19.0],
    [-37.0, 10.0],
    [-7.0, -10.0],
    [-6.0, -23.0],
    [0.0, -30.0],    # close nose tip
], dtype=np.float32)

@njit
def haversine_distance(lat1, lon1, lat2, lon2):
    # Great-circle distance between two lat/lon points (in meters)
    R = 6371000.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1

    a = sin(dφ / 2)**2 + cos(φ1) * cos(φ2) * sin(dλ / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@njit
def distance_3d(lat1, lon1, alt1, lat2, lon2, alt2):
    # Compute 3D distance between observer and aircraft (meters)
    ground_distance = haversine_distance(lat1, lon1, lat2, lon2)
    delta_alt = alt2 - alt1  # usually alt2 is much higher
    return sqrt(ground_distance**2 + delta_alt**2)

@njit
def geodetic_to_enu(lat, lon, alt, obs_lat, obs_lon, obs_alt):
    # Earth radius
    R = 6371000.0
    lat1, lon1 = radians(obs_lat), radians(obs_lon)
    lat2, lon2 = radians(lat), radians(lon)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    x = R * d_lon * cos(lat1)
    y = R * d_lat
    z = alt - obs_alt
    return x, y, z

@njit
def enu_to_az_el(x, y, z):
    horiz_dist = sqrt(x**2 + y**2)
    az = atan2(x, y)
    el = atan2(z, horiz_dist)
    return degrees(az) % 360, degrees(el)

@njit
def az_el_to_pixel(azimuth, elevation, img_size, fov_deg, az_offset_deg):
    if elevation < 0:
        return -1, -1  # below horizon
    adjusted_az = (azimuth - az_offset_deg) % 360.0
    radius = (90.0 - elevation) / (fov_deg / 2.0) * (img_size / 2.0)
    angle_rad = radians(adjusted_az)
    x = img_size / 2.0 + radius * sin(angle_rad)
    y = img_size / 2.0 - radius * cos(angle_rad)
    return int(x), int(y)

def get_newest_index():
    timestamps = metadata_buffer[:, 0]
    newest = np.argmax(timestamps)
    return newest

def draw_plane(img, center, angle_deg, color=(0, 0, 128), scale=1.0):
    angle_rad = np.radians(angle_deg)
    rot_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    pts = PLANE_POLY * scale
    rotated = pts @ rot_mat.T
    translated = rotated + np.array(center)
    pts_int = np.round(translated).astype(np.int32)
    cv2.fillPoly(img, [pts_int], color)

def show_frame(index):
    img = frame_buffer[index]
    debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    scaled_img = debayered_img[::4, ::4] # scaling down to VIEWER_WINDOW = 800
    scaled_img_8bit = cv2.convertScaleAbs(scaled_img, alpha=(255.0/65535.0)) # 8bit for overlays

    if bgs_view:
        mask = mask_buffer[index]
        num_fg_pixels = np.count_nonzero(mask) / mask.size * 100
        mask_resized = cv2.resize(mask, (scaled_img_8bit.shape[1], scaled_img_8bit.shape[0]), interpolation=cv2.INTER_NEAREST)
        scaled_img_8bit[mask_resized != 0] = [0, 0, 255]

    if info_view:
        timestamp = metadata_buffer[index][0]  # float seconds from time.time()
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.') + f"{dt.microsecond // 100:04d} UTC"
        cv2.putText(scaled_img_8bit, f"Index: {index}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(scaled_img_8bit, f"Timestamp: {timestamp_str}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(scaled_img_8bit, f"Exposure: {int(metadata_buffer[index][1])}us", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(scaled_img_8bit, f"Gain: {metadata_buffer[index][2]:.2f}dB", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if bgs_view:
            cv2.putText(scaled_img_8bit, f"Mask pixels: {num_fg_pixels:.4f}%", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)        
    
    if adsb_view:
        aircraft_screen_coords.clear()
        cv2.circle(scaled_img_8bit, center, outer_radius, (0, 0, 128), 1, lineType=cv2.LINE_AA)
        cv2.circle(scaled_img_8bit, center, inner_radius, (0, 0, 128), 1, lineType=cv2.LINE_AA)
        try:
            with open(AIRCRAFT_JSON_PATH, 'r') as f:
                aircraft_data = json.load(f)
            for ac in aircraft_data.get("aircraft", []):
                flight = ac.get("flight", "").strip()
                gs = ac.get("gs", 0)
                if gs is not None: gs *= 1.852 # in km/h
                mag_heading = ac.get("track", 0)
                lat = ac.get("lat")
                lon = ac.get("lon")
                alt = ac.get("alt_baro") or ac.get("alt_geom") or ac.get("alt") # in feet
                if alt is not None: alt *= 0.3048 # in meter
                if lat is None or lon is None or alt is None:
                    continue
                if debug: print(f"lat={lat}, lon={lon}, alt={alt}")
                dist = distance_3d(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, lat, lon, alt)
                x, y, z = geodetic_to_enu(lat, lon, alt, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)
                az, el = enu_to_az_el(x, y, z)
                px, py = az_el_to_pixel(az, el, VIEWER_WINDOW, FOV, AZIMUTH_OFFSET_DEG)
                if 0 <= px < VIEWER_WINDOW and 0 <= py < VIEWER_WINDOW:
                    label = f"{flight}\nalt:{int(alt)} m\ngs:{int(gs)} km/h\nhdg:{int(mag_heading)}\ndist:{(dist // 1000):.1f} km"
                    aircraft_screen_coords.append((px, py, label, mag_heading))
                for px, py, label, heading in aircraft_screen_coords:
                    draw_plane(scaled_img_8bit, (px, py), heading - AZIMUTH_OFFSET_DEG, scale=0.25)
                    # Split label into lines
                    lines = label.split("\n")
                    for i, line in enumerate(lines):
                        cv2.putText(scaled_img_8bit, line, (px + 10, py + 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 128), 1, cv2.LINE_AA)

        except Exception as e:
            print("Aircraft JSON read error:", e)

    cv2.imshow(window_title, scaled_img_8bit)

# Attach to shared memories
shm = shared_memory.SharedMemory(name=RING_BUFFER_NAME)
resource_tracker.unregister(shm._name, 'shared_memory')
frame_buffer = np.ndarray((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=shm.buf)

shm2 = shared_memory.SharedMemory(name=METADATA_BUFFER_NAME)
resource_tracker.unregister(shm2._name, 'shared_memory')
metadata_buffer = np.ndarray((FRAME_COUNT, 3), dtype=np.float64, buffer=shm2.buf)

shm3 = shared_memory.SharedMemory(name=MASK_BUFFER_NAME)
resource_tracker.unregister(shm3._name, 'shared_memory')
mask_buffer = np.ndarray((FRAME_COUNT, FRAME_WIDTH // BGS_DOWNSAMPLE, FRAME_HEIGHT // BGS_DOWNSAMPLE), dtype=np.uint8, buffer=shm3.buf) # BGS runs on 50% size, so 1600x1600

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, VIEWER_WINDOW, VIEWER_WINDOW)

while True:
    key = cv2.waitKey(10) & 0xFF

    if not paused:
        newest_index = (get_newest_index() - 3) % FRAME_COUNT
        if metadata_buffer[newest_index][0] != last_timestamp:
            current_index = newest_index
            last_timestamp = metadata_buffer[current_index][0]
            frame_counter += 1
            show_frame(current_index) # 0.03-0.06s
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
        #print("⏯️ Paused" if paused else "▶️ Resumed")
    elif key == ord('a'):
        adsb_view = not adsb_view
        print(f"ADSB = {adsb_view}")
    elif key == ord('i'):
        info_view = not info_view
        print(f"INFO = {info_view}")
    elif key == ord('b'):
        bgs_view = not bgs_view
        print(f"BGS = {bgs_view}")
    elif key == ord('q'):
        break
    
    time.sleep(0.05)

# Cleanup
cv2.destroyAllWindows()
shm.close()
shm2.close()
shm3.close()
