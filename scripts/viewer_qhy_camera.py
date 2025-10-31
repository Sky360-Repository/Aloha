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
AZIMUTH_OFFSET_DEG = 208
FOV = 160.0

AIRCRAFT_JSON_PATH = os.path.expanduser("~/dump1090-master/dump1090/public_html/data/aircraft.json")
OBSERVER_LAT = 47.647535944954846
OBSERVER_LON = 14.841571479015096
OBSERVER_ALT = 850
MIRROR_EW = True

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
R_corr = 0.85  # (less sensitive in red >> boost it slightly) reference 1.18, testing 0.85
G_corr = 0.85  # reference 1.00, testing 0.85
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

E_POLY = np.array([
    [-7.0, -20.0], #1
    [7.0, -20.0], #2
    [7.0, -17.0], #3
    [-4.0, -17.0], #4
    [-4.0, -13.0], #5
    [4.0, -13.0], #6
    [4.0, -9.0], #7
    [-4.0, -9.0], #8
    [-4.0,  -4.0], #9
    [7.0,  -4.0], #10
    [7.0,  -1.0], #11
    [-7.0, -1.0], #12
], dtype=np.float32)

S_POLY = np.array([
    [7.0, -20.0], #1
    [-4.0, -20.0], #2
    [-7.0, -17.0], #3
    [-7.0, -13.0], #4
    [-4.0, -10.0], #5
    [2.0,  -8.0], #6
    [4.0,  -6.0], #7
    [2.0,  -4.0], #8
    [-7.0, -4.0], #9
    [-7.0, -1.0], #10
    [4.0, -1.0], #11
    [7.0, -4.0], #12
    [7.0, -8.0], #13
    [4.0, -11.0], #14
    [-2.0, -13.0], #15
    [-4.0, -15.0], #16
    [-2.0, -17.0], #17
    [7.0, -17.0], #18
], dtype=np.float32)

W_POLY = np.array([
    [-10.0, -20.0], #1
    [-7.0,  -20.0], #2
    [-5.0,  -5.0], #3
    [0.0,  -14.0], #4
    [5.0, -5.0], #5
    [7.0,  -20.0], #6
    [10.0,  -20.0], #7
    [7.0,  -1.0], #8
    [4.0,  -1.0], #9
    [0.0, -10.0], #10
    [-4.0, -1.0], #11
    [-7.0, -1.0], #12
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
    """
    Bilinear debayer for RGGB Bayer pattern.
    Returns full-resolution RGB (uint16, 0–4095) with simple spectral scaling.
    """
    h, w = raw.shape
    rgb = np.empty((h, w, 3), dtype=np.uint16)

    for y in prange(1, h - 1):
        for x in range(1, w - 1):

            if (y % 2 == 0) and (x % 2 == 0):
                # Red pixel
                R = raw[y, x]
                G = (raw[y, x-1] + raw[y, x+1] + raw[y-1, x] + raw[y+1, x]) * 0.25
                B = (raw[y-1, x-1] + raw[y-1, x+1] + raw[y+1, x-1] + raw[y+1, x+1]) * 0.25

            elif (y % 2 == 0) and (x % 2 == 1):
                # Green pixel on red row
                R = (raw[y, x-1] + raw[y, x+1]) * 0.5
                G = raw[y, x]
                B = (raw[y-1, x] + raw[y+1, x]) * 0.5

            elif (y % 2 == 1) and (x % 2 == 0):
                # Green pixel on blue row
                R = (raw[y-1, x] + raw[y+1, x]) * 0.5
                G = raw[y, x]
                B = (raw[y, x-1] + raw[y, x+1]) * 0.5

            else:
                # Blue pixel
                R = (raw[y-1, x-1] + raw[y-1, x+1] + raw[y+1, x-1] + raw[y+1, x+1]) * 0.25
                G = (raw[y, x-1] + raw[y, x+1] + raw[y-1, x] + raw[y+1, x]) * 0.25
                B = raw[y, x]

            # --- simple spectral sensitivity compensation ---
            R = min(R * R_corr, 4095.0)
            G = min(G * G_corr, 4095.0)
            B = min(B * B_corr, 4095.0)

            rgb[y, x, 0] = np.uint16(R)
            rgb[y, x, 1] = np.uint16(G)
            rgb[y, x, 2] = np.uint16(B)

    # optional: copy border pixels from neighbors
    for x in prange(w):
        rgb[0, x] = rgb[1, x]
        rgb[h - 1, x] = rgb[h - 2, x]
    for y in prange(h):
        rgb[y, 0] = rgb[y, 1]
        rgb[y, w - 1] = rgb[y, w - 2]

    return rgb


def draw_cardinal(img, poly, target_az_deg, az_offset_deg, scale=2.0, color=(0,0,128)):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(h * 0.48)

    p = poly.astype(np.float32) * scale
    poly_centered = p + np.array([cx, cy], dtype=np.float32)

    global MIRROR_EW
    sign = -1 if not MIRROR_EW else +1
    #theta = radians(sign * (az_offset_deg + target_az_deg))
    theta = radians(sign * (az_offset_deg + target_az_deg - 90.0))

    rot_mat = np.array([[cos(theta), -sin(theta)],
                        [sin(theta),  cos(theta)]], dtype=np.float32)

    rotated = (poly_centered - np.array([cx, cy])) @ rot_mat.T + np.array([cx, cy])

    angle_rad = theta
    dx = radius * sin(angle_rad)
    dy = -radius * cos(angle_rad)
    translated = rotated + np.array([dx, dy], dtype=np.float32)

    pts = translated.astype(np.int32)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)


def heading_to_compass(heading_deg):
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((heading_deg + 11.25) / 22.5) % 16
    return directions[idx]


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


def draw_text_box(img, text, org, font_scale=1, color=(0, 0, 128), thickness=2):
    lines = text.split("\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(36 * font_scale)
    max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines])
    x, y = org
    for i, line in enumerate(lines):
        text_y = y + (i + 1) * line_height
        cv2.putText(img, line, (x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


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
        img_8bit[mask_resized != 0] = [0, 0, 200]

    if info_view:
        timestamp = metadata_buffer[index][0]  # float seconds from time.time()
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.') + f"{dt.microsecond // 100:04d} UTC"
        
        # Data
        cv2.putText(img_8bit, f"Index: {index}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Timestamp: {timestamp_str}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Exposure: {int(metadata_buffer[index][1])}us", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gain: {metadata_buffer[index][2]:.2f}dB", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gamma: {gamma:.1f}", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        if bgs_view: cv2.putText(img_8bit, f"Mask pixels: {num_fg_pixels:.4f}%", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)

        # Keys
        cv2.putText(img_8bit, "a ...ADSB", (40, 2820), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "b ...BGS", (40, 2860), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "h ...humanRGB", (40, 2900), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "+ ...Gamma++", (40, 2940), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "- ...Gamma--", (40, 2980), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "2 ...pause", (40, 3020), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "3 ...forward", (40, 3060), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "1 ...backward", (40, 3100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "q ...quit", (40, 3140), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        
        # Flags
        if humanRGB: cv2.putText(img_8bit, "humanRGB", (1450,3140), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)

    if adsb_view:
        aircraft_screen_coords.clear()
        cv2.circle(img_8bit, center, outer_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA)
        cv2.circle(img_8bit, center, inner_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA)

        # --- draw cardinal letters N, E, S, W (respect MIRROR_EW for E/W) ---
        # N: 0 deg, E: 90 deg, S: 180 deg, W: 270 deg
        cardinal_list = [
            (NORTH_POLY, 0.0),
            (E_POLY, 90.0),
            (S_POLY, 180.0),
            (W_POLY, 270.0),
        ]
        for poly, az_deg in cardinal_list:
            draw_az = az_deg
            # If the fisheye is mirrored left-right, swap E <-> W by mirroring their azimuth
            if MIRROR_EW and (az_deg == 90.0 or az_deg == 270.0):
                draw_az = (360.0 - az_deg) % 360.0
            draw_cardinal(img_8bit, poly, draw_az, AZIMUTH_OFFSET_DEG, scale=2.0, color=(0,0,128))

        # Read aircraft JSON file
        try:
            with open(AIRCRAFT_JSON_PATH, 'r') as f:
                aircraft_data = json.load(f)

            for ac in aircraft_data.get("aircraft", []):
                flight = ac.get("flight", "").strip()
                gs = ac.get("gs", 0)
                if gs is not None:
                    gs *= 1.852  # convert to km/h

                mag_heading = ac.get("track", 0)
                lat = ac.get("lat")
                lon = ac.get("lon")
                alt = ac.get("alt_baro") or ac.get("alt_geom") or ac.get("alt")  # in feet
                if alt is not None:
                    alt *= 0.3048  # to meters

                if lat is None or lon is None or alt is None:
                    continue

                if adsb_debug: print(f"lat={lat}, lon={lon}, alt={alt}")

                dist = distance_3d(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, lat, lon, alt)
                x, y, z = geodetic_to_enu(lat, lon, alt, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)
                az, el = enu_to_az_el(x, y, z)

                # --- Mirror East/West if fisheye image is mirrored left-right ---
                if MIRROR_EW: az = (360.0 - az) % 360.0

                px, py = az_el_to_pixel(az, el, VIEWER_WINDOW, FOV, AZIMUTH_OFFSET_DEG)

                if 0 <= px < VIEWER_WINDOW and 0 <= py < VIEWER_WINDOW:
                    # Mirror heading as well so plane points correctly
                    heading = mag_heading if mag_heading is not None else 0.0
                    if MIRROR_EW: heading = (360.0 - heading) % 360.0

                    label = (
                        f"{flight}\n"
                        f"alt:{int(alt)} m\n"
                        f"gs:{int(gs)} km/h\n"
                        f"hdg:{int(mag_heading)} {heading_to_compass(mag_heading)}\n"
                        f"dist:{(dist // 1000):.1f} km"
                    )

                    aircraft_screen_coords.append((px, py, label, heading))

        except Exception as e:
            print("Aircraft JSON read error:", e)

        # Draw aircraft
        for px, py, label, heading in aircraft_screen_coords:
            draw_plane(img_8bit, (px, py), heading - AZIMUTH_OFFSET_DEG, scale=0.5)
            # Better readable text box
            draw_text_box(img_8bit, label, (px + 10, py + 20))

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
    if paused:
        # Blocking wait for key (never misses input)
        key = cv2.waitKey(0) & 0xFF
    else:
        key = cv2.waitKey(10) & 0xFF  # non-blocking short wait

    # --- global toggles ---
    if key == ord('2'):
        paused = not paused
        print("Paused =", paused)
        if paused:
            show_frame(current_index)
        continue

    if key == ord('a'):
        adsb_view = not adsb_view
        print(f"ADSB = {adsb_view}")
        show_frame(current_index)
        continue

    if key == ord('b'):
        bgs_view = not bgs_view
        print(f"BGS = {bgs_view}")
        show_frame(current_index)
        continue

    if key == ord('h'):
        humanRGB = not humanRGB
        print(f"humanRGB = {humanRGB}")
        show_frame(current_index)
        continue

    if key == ord('i'):
        info_view = not info_view
        print(f"INFO = {info_view}")
        show_frame(current_index)
        continue

    if key == ord('+'):
        gamma = min(GAMMA_MAX, gamma + GAMMA_INC)
        print(f"Gamma increased to {gamma:.1f}")
        show_frame(current_index)
        continue

    if key == ord('-'):
        gamma = max(GAMMA_MIN, gamma - GAMMA_INC)
        print(f"Gamma decreased to {gamma:.1f}")
        show_frame(current_index)
        continue

    if key == ord('q'):
        break

    # --- playback control ---
    if paused:
        # Step through frames manually
        if key == ord('3'):
            current_index = (current_index + 1) % FRAME_COUNT
            show_frame(current_index)
        elif key == ord('1'):
            current_index = (current_index - 1) % FRAME_COUNT
            show_frame(current_index)
    else:
        # Live mode (follow newest frames)
        newest_index = (get_newest_index() - 3) % FRAME_COUNT
        if metadata_buffer[newest_index][0] != last_timestamp:
            current_index = newest_index
            last_timestamp = metadata_buffer[current_index][0]
            frame_counter += 1
            show_frame(current_index)


# Cleanup
cv2.destroyAllWindows()
shm.close()
shm2.close()
shm3.close()
