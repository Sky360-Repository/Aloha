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
from numba import njit, prange
from math import radians, degrees, sin, cos, atan2, sqrt, asin

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"
MASK_BUFFER_NAME = "mask_ring_buffer"
FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200
BGS_DOWNSAMPLE = 2
VIEWER_WINDOW = 3200
AZIMUTH_OFFSET_DEG = 220
FOV = 160.0

AIRCRAFT_JSON_PATH = os.path.expanduser("~/dump1090-master/dump1090/public_html/data/aircraft.json")
OBSERVER_LAT = 47.647535944954846
OBSERVER_LON = 14.841571479015096
OBSERVER_ALT = 850

GAMMA_MIN = 0.1
GAMMA_MAX = 3.0
GAMMA_INC = 0.1
gamma = 2.2

FONT_SIZE = 1.2

debug = True
adsb_debug = False
aircraft_screen_coords = []

# Viewer states
last_timestamp = 0.0
current_index = 0
frame_counter = 0
fps_timer_start = time.time()
window_title = "Sky360 Fisheye Viewer (i...info)"
paused = False
bgs_view = False
info_view = False
adsb_view = False
humanRGB = False

center = (VIEWER_WINDOW // 2, VIEWER_WINDOW // 2)
outer_radius = VIEWER_WINDOW // 2
inner_radius = int((90.0 / FOV) * outer_radius)

scale_prev = np.ones(3, dtype=np.float32)
alpha = 0.1  # EMA smoothing

# --- Spectral correction coefficients (from fitted sensitivity balance)
# Normalize such that average gain = 1.0
R_corr = 1.18  # less sensitive in red, boost it slightly
G_corr = 1.00  # reference
B_corr = 1.32  # more loss in blue channel

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

NORTH_POLY = np.array([
    [0.0, -27.0], #1
    [-5.0, -24.0], #2
    [-8.0, -24.0], #3
    [0.0, -30.0], #4
    [8.0, -24.0], #5
    [5.0, -24.0], #6
    [0.0, -27.0], #7
    [0.0, -12.0], #8
    [-5.0, -20.0], #9
    [-7.0, -20.0], #10
    [-7.0, -1.0], #11
    [-5.0, -1.0], #12
    [-5.0, -16.0], #13
    [5.0, -1.0], #14
    [7.0, -1.0], #15
    [7.0, -20.0], #16
    [5.0, -20.0], #17
    [5.0, -5.0], #18
    [0.0, -12.0], #19
    [0.0, -27.0], #20
], dtype=np.float32)

# --- Gamma + scale LUT for ultra-fast conversion ---
class GammaScaler:
    def __init__(self, gamma=2.2, scale=None):
        self.gamma = gamma
        self.inv_gamma = 1.0 / gamma
        if scale is None:
            self.scale = np.ones(3, dtype=np.float32)
        else:
            self.scale = np.array(scale, dtype=np.float32)
        self.lut = self._make_lut()

    def _make_lut(self):
        lut = np.zeros((3, 4096), dtype=np.uint8)
        for c in range(3):
            for i in range(4096):
                val = ((i / 4095.0) ** self.inv_gamma) * self.scale[c]
                val = min(max(val, 0.0), 1.0)
                lut[c, i] = np.uint8(val * 255 + 0.5)
        return lut

    def update_gamma_scale(self, gamma=None, scale=None):
        updated = False
        if gamma is not None and gamma != self.gamma:
            self.gamma = gamma
            self.inv_gamma = 1.0 / gamma
            updated = True
        if scale is not None and not np.allclose(scale, self.scale):
            self.scale = np.array(scale, dtype=np.float32)
            updated = True
        if updated:
            self.lut = self._make_lut()

    def apply(self, img_uint16):
        return gamma_and_scale_lut(img_uint16, self.lut)

gamma_scaler = GammaScaler(gamma=gamma, scale=scale_prev)

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
    # Adjust azimuth by camera offset (positive means the camera image is rotated clockwise relative to true north)
    adjusted_az = (azimuth - az_offset_deg) % 360.0
    radius = (90.0 - elevation) / (fov_deg / 2.0) * (img_size / 2.0)
    half = img_size / 2.0
    if radius > half:
        radius = half
    angle_rad = radians(adjusted_az)
    x = half + radius * sin(angle_rad)
    y = half - radius * cos(angle_rad)
    return int(round(x)), int(round(y))
         
@njit(parallel=True, fastmath=True)
def gamma_and_scale_lut(img_uint16, lut):
    H, W, C = img_uint16.shape
    out = np.empty((H, W, C), dtype=np.uint8)
    for c in prange(C):
        for y in range(H):
            for x in range(W):
                out[y, x, c] = lut[c, img_uint16[y, x, c]]
    return out

@njit(parallel=True, fastmath=True)
def debayer_RGGB2humanRGB(raw):
    h, w = raw.shape
    rgb = np.empty((h, w, 3), dtype=np.uint16)
    h2, w2 = int(h // 2), int(w // 2)

    for y2 in prange(h2):
        y = y2 * 2
        for x2 in range(w2):
            x = x2 * 2

            # read RGGB 2x2 block
            r = raw[y, x]
            g1 = raw[y, x + 1]
            g2 = raw[y + 1, x]
            b = raw[y + 1, x + 1]
            g = (g1 + g2) * 0.5

            # clamp to 12-bit and scale
            r_val = min(int(r * 1.18), 4095)
            g_val = min(int(g * 1.00), 4095)
            b_val = min(int(b * 1.32), 4095)

            # fill 2x2 block
            rgb[y, x, 0] = r_val
            rgb[y, x, 1] = g_val
            rgb[y, x, 2] = b_val

            rgb[y, x + 1, 0] = r_val
            rgb[y, x + 1, 1] = g_val
            rgb[y, x + 1, 2] = b_val

            rgb[y + 1, x, 0] = r_val
            rgb[y + 1, x, 1] = g_val
            rgb[y + 1, x, 2] = b_val

            rgb[y + 1, x + 1, 0] = r_val
            rgb[y + 1, x + 1, 1] = g_val
            rgb[y + 1, x + 1, 2] = b_val

    return rgb



def draw_north_marker(img, az_offset_deg, scale=2.0, thickness=2):
    color = (0, 0, 128)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(h * 0.48)  # distance from image center

    # Scale polygon
    poly = NORTH_POLY.astype(np.float32) * scale

    # Translate polygon to image center (so rotation occurs around image center)
    poly_centered = poly + np.array([cx, cy], dtype=np.float32)

    # Rotation matrix around center
    theta = radians(-az_offset_deg)  # clockwise
    rot_mat = np.array([[cos(theta), -sin(theta)],
                        [sin(theta),  cos(theta)]], dtype=np.float32)

    # Rotate each point around center
    rotated = (poly_centered - np.array([cx, cy])) @ rot_mat.T + np.array([cx, cy])

    # Now translate along radial to circle edge
    angle_rad = radians(-az_offset_deg)
    dx = radius * sin(angle_rad)
    dy = -radius * cos(angle_rad)
    translated = rotated + np.array([dx, dy], dtype=np.float32)

    # Draw polygon
    pts = translated.astype(np.int32)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    #cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

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
    global_time = time.perf_counter()
    global scale_prev
    img = frame_buffer[index]
    if debug: print(f"Index: {index} ------------------------------------")
    
    ### Image preprocess for viewing
    
    # Debayering - 0.0225s
    start_time = time.perf_counter()
    if humanRGB:
        debayered_img = debayer_RGGB2humanRGB(img)
    else:
        debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRGGB2RGB_EA)
    if debug: print(f"compute debayering: {(time.perf_counter() - start_time):.4f}s")
    
    # Compute per-channel average on downsampled 8-bit for EMA - 
    start_time = time.perf_counter()
    avg_down = debayered_img[::4, ::4, :].mean(axis=(0,1)).astype(np.float32)
    gray = avg_down.mean()
    scale = alpha * (gray / avg_down) + (1 - alpha) * scale_prev
    scale_prev = scale.copy()
    if debug: print(f"compute EMA scale: {(time.perf_counter() - start_time):.4f}s")
    
    # Update LUT if Gamma or per-channel scale changed
    start_time = time.perf_counter()
    gamma_scaler.update_gamma_scale(gamma=gamma, scale=scale)
    img_8bit = gamma_scaler.apply(debayered_img)
    if debug: print(f"compute Gamma: {(time.perf_counter() - start_time):.4f}s")

    ### UI

    if bgs_view:
        mask = mask_buffer[index]
        num_fg_pixels = np.count_nonzero(mask) / mask.size * 100
        mask_resized = cv2.resize(mask, (img_8bit.shape[1], img_8bit.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_8bit[mask_resized != 0] = [0, 0, 255]

    if info_view:
        timestamp = metadata_buffer[index][0]  # float seconds from time.time()
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.') + f"{dt.microsecond // 100:04d} UTC"
        cv2.putText(img_8bit, f"Index: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Timestamp: {timestamp_str}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Exposure: {int(metadata_buffer[index][1])}us", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gain: {metadata_buffer[index][2]:.2f}dB", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gamma: {gamma:.1f} (press +/-)", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        if bgs_view: cv2.putText(img_8bit, f"Mask pixels: {num_fg_pixels:.4f}%", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "Keys:", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "a ...ADSB", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "b ...BGS", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "2 ...pause", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "3 ...forward", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "1 ...backward", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.putText(img_8bit, "q ...quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 2, cv2.LINE_AA)

    if adsb_view:
        aircraft_screen_coords.clear()
        cv2.circle(img_8bit, center, outer_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA)
        cv2.circle(img_8bit, center, inner_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA)
        draw_north_marker(img_8bit, AZIMUTH_OFFSET_DEG, scale=2.0, thickness=4)
        # Read aircraft JSON file
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
                if adsb_debug: print(f"lat={lat}, lon={lon}, alt={alt}")
                dist = distance_3d(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, lat, lon, alt)
                x, y, z = geodetic_to_enu(lat, lon, alt, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)
                az, el = enu_to_az_el(x, y, z)
                px, py = az_el_to_pixel(az, el, VIEWER_WINDOW, FOV, AZIMUTH_OFFSET_DEG)
                if 0 <= px < VIEWER_WINDOW and 0 <= py < VIEWER_WINDOW:
                    label = f"{flight}\nalt:{int(alt)} m\ngs:{int(gs)} km/h\nhdg:{int(mag_heading)}\ndist:{(dist // 1000):.1f} km"
                    aircraft_screen_coords.append((px, py, label, mag_heading))
        except Exception as e:
            print("Aircraft JSON read error:", e)
        # 
        for px, py, label, heading in aircraft_screen_coords:
            draw_plane(img_8bit, (px, py), heading - AZIMUTH_OFFSET_DEG, scale=0.25)
            # Split label into lines
            lines = label.split("\n")
            for i, line in enumerate(lines):
                cv2.putText(img_8bit, line, (px + 10, py + 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 128), 1, cv2.LINE_AA)

    cv2.imshow(window_title, img_8bit)
    if debug: print(f"total compute FPS: {1 / (time.perf_counter() - global_time):.2f}")

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
cv2.resizeWindow(window_title, 1080, 1080)

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
        if key == ord('3') or key == 83:  # Right arrow or 'd'
            current_index = (current_index + 1) % FRAME_COUNT
            show_frame(current_index)
        elif key == ord('1') or key == 81:  # Left arrow or 'a'
            current_index = (current_index - 1) % FRAME_COUNT
            show_frame(current_index)

    # Handle toggle and quit
    if key == ord('2'):
        paused = not paused
    elif key == ord('a'):
        adsb_view = not adsb_view
        print(f"ADSB = {adsb_view}")
    elif key == ord('h'):
        humanRGB = not humanRGB
        print(f"humanRGB = {humanRGB}")
    elif key == ord('i'):
        info_view = not info_view
        print(f"INFO = {info_view}")
    elif key == ord('+'):
        gamma = min(GAMMA_MAX, gamma + GAMMA_INC)
        print(f"Gamma increased to {gamma:.1f}")
    elif key == ord('-'):
        gamma = max(GAMMA_MIN, gamma - GAMMA_INC)
        print(f"Gamma decreased to {gamma:.1f}")
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
