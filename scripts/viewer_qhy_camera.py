#!/usr/bin/env python3
# coding: utf-8

import time
from datetime import datetime, timezone
import numpy as np
import math
import cv2
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as resource_tracker
import json
import os
from numba import njit, prange
from math import radians, degrees, sin, cos, atan2, sqrt, asin, acos, hypot
from scipy.optimize import least_squares
from collections import deque
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, solar_system_ephemeris, get_body

import astropy.units as u

# Constants (must match QHYCameraController config)
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"
MASK_BUFFER_NAME = "mask_ring_buffer"
FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200
BGS_DOWNSAMPLE = 2
VIEWER_WINDOW = 3200
AZIMUTH_OFFSET_DEG = 215.4910559206
FOV = 120.0
FISHEYE_CURVATURE = 0.7 # 1.0...spherical, <1.0...flat, >1.0...squeezed
half = VIEWER_WINDOW / 2.0

# Sensor/optics misalignment
TILT_COMP_N = 0.95  # degrees, compensation for tilting the optical axis N to S (not horizontally levelled bed)
TILT_COMP_W = 0.50  # degrees, compensation for tilting the optical axis W to E (not horizontally levelled bed)
DELTA_X = -5 # x px, compensation for sensor/optic misalignment
DELTA_Y = 13 # y px, compensation for sensor/optic misalignment
# coefficients found by stars mapping and posting JSON file to ChatGTP for a fit curve
#coeffs = (0.0006, 0.9306, 0.0897, -0.0204) 
#coeffs = (0.0006, 0.9452, 0.0897, -0.0204)
#coeffs = (0.0006, 0.9608, 0.0897, -0.0204)
#coeffs = (0.0006, 0.9750, 0.0897, -0.0204)
#coeffs = (0.0006, 0.9896, 0.0897, -0.0204)
#coeffs = (0.0006, 1.0040, 0.0897, -0.0204)
#coeffs = (0.0006, 1.0185, 0.0897, -0.0204)
#coeffs = (0.0006, 1.012, 0.0897, -0.0204)
#coeffs = (0.0531832460, 1.00273127, 0.111288099, -0.0384216483)
#coeffs = (0.0596583309, 0.9744371819, 0.157787811, -0.0626728428)
#coeffs = (0.0485, 0.9987, 0.1132, -0.0368)
#coeffs = (0.0512, 1.0005, 0.1154, -0.0379)
#coeffs = (0.0521, 0.9998, 0.1163, -0.0374)
#coeffs = (0.0521, 1.002, 0.1163, -0.0374)
#coeffs = (0.0521, 1.004, 0.1163, -0.0374)
#coeffs = (0.049, 0.998, 0.114, -0.031)
#coeffs = (0.049, 0.996, 0.115, -0.028)
#coeffs = (0.050, 0.995, 0.112, -0.026)
#coeffs = (0.051, 0.994, 0.110, -0.025)
#coeffs = (0.052, 0.992, 0.115, -0.028)
coeffs = (0.0357361073520546, 0.9880883959579535, 0.11702050512672706, -0.026105839115683687)

# ADSB config
AIRCRAFT_JSON_PATH = os.path.expanduser("~/dump1090-master/dump1090/public_html/data/aircraft.json")
OBSERVER_LAT = 47.647535944954846
OBSERVER_LON = 14.841571479015096
OBSERVER_ALT = 850
MIRROR_EW = True

# ADSB track config
TRACK_MAX_LEN = 200 # max points per track
TRACK_TTL = 10.0 # seconds after last seen to delete the track
MIN_MOVE_PIXELS = 4 # minimum pixel movement to append a new point

GAMMA_MIN = 0.1
GAMMA_MAX = 3.0
GAMMA_INC = 0.1
gamma = 2.2

FONT_SIZE = 1.4

debug = True
adsb_debug = False

aircraft_screen_coords = []
tracks = {} # mapping icao -> {'points': deque, 'last_seen': float, 'label': str}

# Viewer states
last_timestamp = 0.0
current_index = 0
frame_counter = 0
fps_timer_start = time.time()
window_title = "Sky360 Fisheye Viewer (i...info)"
paused = False
bgs_view = False
info_view = True
cardinals_view = True
adsb_view = False
humaneRGB = False
stars_view = False

# --- Star calibration mode ---
global star_points
star_points = []
global planet_points
planet_points = []
global expected_star_pixels
expected_star_pixels = []
global selected_star
selected_star = -1

MAG_LEVEL = 4.0
MAG_INC = 0.5

scale_prev = np.ones(3, dtype=np.float32)
alpha = 0.1  # EMA smoothing

# --- Spectral correction coefficients (from fitted sensitivity balance)
# Normalize such that average gain = 1.0
R_corr = 0.85  # (less sensitive in red >> boost it slightly) reference 1.18, testing 0.85
G_corr = 0.85  # reference 1.00, testing 0.85
B_corr = 1.32  # more loss in blue channel

# A compact bright‐star list (RA in hours, Dec in degrees)
BRIGHT_STARS = [
    ("Sirius",        6.7525,  -16.7161, -1.46),
    ("Canopus",       6.3992,  -52.6957, -0.72),
    ("Rigil Kent",   14.6601,  -60.8356, -0.27),  # Alpha Centauri
    ("Arcturus",     14.2610,  +19.1825, -0.05),
    ("Vega",         18.6156,  +38.7837,  0.03),
    ("Capella",       5.2782,  +45.9980,  0.08),
    ("Rigel",         5.2423,   -8.2016,  0.12),
    ("Procyon",       7.6550,   +5.2250,  0.38),
    ("Achernar",      1.6286,  -57.2367,  0.46),
    ("Betelgeuse",    5.9195,   +7.4071,  0.50),
    ("Hadar",        14.0637,  -60.3730,  0.61),
    ("Altair",       19.8464,   +8.8683,  0.77),
    ("Acrux",        12.4433,  -63.0991,  0.77),
    ("Aldebaran",     4.5987,  +16.5093,  0.85),
    ("Antares",      16.4901,  -26.4319,  0.96),
    ("Spica",        13.4199,  -11.1613,  0.98),
    ("Pollux",        7.7553,  +28.0262,  1.14),
    ("Fomalhaut",    22.9608,  -29.6222,  1.16),
    ("Deneb",        20.6905,  +45.2803,  1.25),
    ("Mimosa",       12.7953,  -59.6888,  1.25),
    ("Regulus",      10.1395,  +11.9672,  1.35),
    ("Adhara",        6.9771,  -28.9721,  1.50),
    ("Shaula",       17.5601,  -37.1038,  1.62),
    ("Castor",        7.5766,  +31.8883,  1.58),
    ("Gacrux",       12.5194,  -57.1132,  1.63),
    ("Bellatrix",     5.4189,   +6.3497,  1.64),
    ("Elnath",        5.4382,  +28.6074,  1.65),
    ("Miaplacidus",   9.2199,  -69.7172,  1.67),
    ("Alnair",       22.1372,  -46.9606,  1.73),
    ("Alioth",       12.9004,  +55.9598,  1.76),
    ("Alkaid",       13.7923,  +49.3133,  1.85),
    ("Dubhe",        11.0621,  +61.7508,  1.79),
    ("Mirfak",        3.4054,  +49.8612,  1.79),
    ("Wezen",         7.1399,  -26.3932,  1.83),
    ("Sargas",       17.6228,  -43.9870,  1.86),
    ("Kaus Australis",18.4029, -34.3846,  1.79),
    ("Avior",         8.3752,  -59.5095,  1.86),
    ("Alhena",        6.6285,  +16.3993,  1.93),
    ("Peacock",      20.4275,  -56.7351,  1.94),
    ("Alsephina",     7.2857,  -29.3031,  1.98),
    ("Menkalinan",    5.9921,  +44.9474,  1.90),
    ("Atria",        16.8111,  -69.0277,  1.91),
    ("Kochab",       14.8451,  +74.1555,  2.07),
    ("Alnitak",       5.6793,   -1.9426,  1.74),
    ("Saiph",         5.7959,   -9.6696,  2.06),
    ("Diphda",        0.7265,  -17.9866,  2.04),
    ("Eltanin",      17.9434,  +51.4889,  2.24),
    ("Denebola",     11.8177,  +14.5721,  2.14),
    ("Enif",         21.7364,   +9.8750,  2.38),
    ("Aspidiske",     9.2848,  -59.2752,  1.68),
    ("Algol",         3.1361,  +40.9556,  2.12),
    ("Mirzam",        6.3783,  -17.9559,  1.99),
    ("Polaris",       2.5303,  +89.2641,  1.97),
    ("Almaaz",        5.9920,  +43.8233,  2.90),
    ("Aludra",        7.4016,  -29.3031,  2.45),
    ("Markab",       23.0793,  +15.2053,  2.49),
    ("Rasalhague",   17.5822,  +12.5600,  2.08),
    ("Nunki",        18.9211,  -26.2967,  2.05),
    ("Alpheratz",     0.1398,  +29.0904,  2.06),
    ("Scheat",       23.0629,  +28.0828,  2.43),
    ("Algenib",       0.2206,  +15.1836,  2.83),
    ("Caph",          0.1529,  +59.1498,  2.28),
    ("Rukbat",       19.3980,  -40.6163,  3.97),
    ("Menkar",        3.0379,   +4.0897,  2.54),
    ("Zubenelgenubi",14.8479,  -16.0418,  2.75),
    ("Sadalmelik",   22.0964,   -0.3199,  2.96),
    ("Sadalsuud",    21.5259,   -5.5712,  2.87),
]


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

ZENITH_POLY = np.array([
    [-1.0, 1.0],
    [-1.0, 8.0],
    [1.0, 8.0],
    [1.0, 1.0],
    [8.0, 1.0],
    [8.0, -1.0],
    [1.0, -1.0],
    [1.0, -8.0],
    [-1.0, -8.0],
    [-1.0, -1.0],
    [-8.0, -1.0],
    [-8.0, 1.0],
    [-1.0, 1.0],
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


class FisheyeRefitter:
    def __init__(self, star_points, expected_star_pixels, cx_cy, Rrim_px, fov_deg):
        """
        star_points: list of measured stars [(name, x, y, mag, az, el), ...]
        expected_star_pixels: list of target positions [(name, x, y, ...), ...]
        cx_cy: camera center [cx, cy]
        Rrim_px: radius of fisheye circle
        fov_deg: camera FOV
        """
        self.star_points = star_points
        self.expected_star_pixels = expected_star_pixels
        self.cx, self.cy = cx_cy
        self.Rrim_px = Rrim_px
        self.fov_deg = fov_deg

        # Initialize parameters from current values
        self.tilt_N_W = np.array([0.0, 0.0])
        self.delta_X_Y = np.array([0.0, 0.0])
        self.coeffs = np.array([0.0, 1.0, 0.0, 0.0])  # example 4th-order poly

    # --- Forward projection using current params ---
    def project_star(self, az, el, tilt_N_W, delta_X_Y, coeffs):
        """
        az, el in degrees
        Returns x, y in pixels
        """
        # Convert to radians
        az_rad = np.radians(az)
        el_rad = np.radians(el)

        # Apply tilt
        el_tilted = el_rad + np.radians(tilt_N_W[0])
        az_tilted = az_rad + np.radians(tilt_N_W[1])

        # Convert to normalized radius (0..1)
        r = (np.pi/2 - el_tilted) / np.radians(self.fov_deg/2)
        r = coeffs[0] + coeffs[1]*r + coeffs[2]*r**2 + coeffs[3]*r**3
        r_px = r * self.Rrim_px

        # Pixel coordinates
        x = self.cx + r_px * np.sin(az_tilted) + delta_X_Y[0]
        y = self.cy - r_px * np.cos(az_tilted) + delta_X_Y[1]

        return x, y

    # --- Residuals for least_squares ---
    def residuals_tilt_delta(self, params):
        tilt_N_W = params[:2]
        delta_X_Y = params[2:4]
        res = []
        for meas, exp in zip(self.star_points, self.expected_star_pixels):
            x_proj, y_proj = self.project_star(meas[4], meas[5], tilt_N_W, delta_X_Y, self.coeffs)
            res.append(x_proj - exp[1])
            res.append(y_proj - exp[2])
        return np.array(res)

    def residuals_coeffs(self, params):
        coeffs = params
        res = []
        for meas, exp in zip(self.star_points, self.expected_star_pixels):
            x_proj, y_proj = self.project_star(meas[4], meas[5], self.tilt_N_W, self.delta_X_Y, coeffs)
            res.append(x_proj - exp[1])
            res.append(y_proj - exp[2])
        return np.array(res)

    # --- Fit tilt/delta first ---
    def fit_tilt_delta(self):
        x0 = np.concatenate([self.tilt_N_W, self.delta_X_Y])
        result = least_squares(self.residuals_tilt_delta, x0, verbose=1)
        self.tilt_N_W = result.x[:2]
        self.delta_X_Y = result.x[2:4]
        return result

    # --- Fit polynomial coefficients next ---
    def fit_coeffs(self):
        x0 = self.coeffs
        result = least_squares(self.residuals_coeffs, x0, verbose=1)
        self.coeffs = result.x
        return result

    # --- Compute total pixel error ---
    def total_error(self):
        res = self.residuals_tilt_delta(np.concatenate([self.tilt_N_W, self.delta_X_Y]))
        return np.mean(np.abs(res))

    # --- Run full refit ---
    def refit(self):
        if debug: print("Fitting tilt/deltas...")
        self.fit_tilt_delta()
        if debug: print(f"Updated tilt/delta: {self.tilt_N_W}, {self.delta_X_Y}")
        if debug: print("Fitting polynomial coefficients...")
        self.fit_coeffs()
        if debug: print(f"Updated coeffs: {self.coeffs}")
        if debug: print(f"Total pixel error: {self.total_error():.2f}")
        return {
            "tilt_N_W": self.tilt_N_W.tolist(),
            "delta_X_Y": self.delta_X_Y.tolist(),
            "coeffs": self.coeffs.tolist(),
            "error_px": self.total_error()
        }


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
    # x = east, y = north, z = up
    # azimuth: 0°=north, 90°=east
    az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    horizontal_dist = np.sqrt(x**2 + y**2)
    el = np.degrees(np.arctan2(z, horizontal_dist))
    return az, el


@njit
def radius_for_elevation_poly(elev_deg, Rrim_px, fov_deg, coeffs):
    """
    Polynomial mapping from elevation to pixel radius.

    r = Rrim_px * (a0 + a1*u + a2*u^2 + a3*u^3 + ...)
    where u = sin((90 - elev)/2) / sin(fov/2)

    Args:
        elev_deg: Elevation angle (degrees)
        Rrim_px:  Pixel radius of horizon in image (px)
        fov_deg:  Full field of view (degrees)
        coeffs:   Polynomial coefficients [a0, a1, a2, ...] (list or tuple)
    """
    theta = math.radians(90.0 - elev_deg)
    theta_max = math.radians(fov_deg / 2.0)
    denom = math.sin(theta_max / 2.0)
    if denom == 0.0:
        return 0.0

    u = math.sin(theta / 2.0) / denom
    if u < 0.0:
        u = 0.0

    # Polynom
    poly = 0.0
    power = 1.0
    for i in range(len(coeffs)):
        poly += coeffs[i] * power
        power *= u

    r = Rrim_px * poly
    return r


@njit
def radius_for_elevation_equisolid(elev_deg, Rrim_px, fov_deg, p):
    """
    Generalized equisolid with exponent p:
    r = Rrim * ( sin((90-elev)/2) / sin(fov/2) ) ** p
    - p = 1.0 -> standard equisolid
    where:
      theta = radians(90 - elev_deg)        (angular distance from zenith)
      theta_max = radians(fov_deg / 2)      (max angular distance from zenith)
      Rrim_px = pixel radius measured for the rim/horizon in the image (px)
      fov_deg = full FOV in degrees
    """
    theta = math.radians(90.0 - elev_deg)
    theta_max = math.radians(fov_deg / 2.0)

    denom = math.sin(theta_max / 2.0)
    # avoid division by zero
    if denom == 0.0:
        return 0.0

    u = math.sin(theta / 2.0) / denom
    
    # clamp u for numerical stability, but allow >1 for out-of-FOV
    u = max(0.0, u)
    
    # power-law shape factor
    r = Rrim_px * (u ** p)
    return r
    

@njit
def radius_for_elevation_fisheye(elev_deg, img_size=VIEWER_WINDOW, fov_deg=FOV):
    """
    Fisheye equidistant projection: r = f * theta
    Elevation = 0° (horizon) -> outer edge
    Elevation = 90° (zenith) -> center
    """
    half_px = half
    theta_max = np.radians(fov_deg / 2)  # max angular distance from zenith
    theta = np.radians(90.0 - elev_deg)  # angular distance from zenith

    # clamp theta
    theta = min(theta, theta_max)

    # focal length in pixels
    f = half_px / theta_max

    r = f * theta
    return r


@njit
def radius_for_elevation_linear(elev_deg, img_size=VIEWER_WINDOW, fov_deg=FOV):
    """
    Linear mapping from elevation to pixel radius.
    Elevation = 0° (horizon) -> outer edge
    Elevation = 90° (zenith)  -> image center
    img_size = VIEWER_WINDOW
    fov_deg = camera FOV
    """
    half_px = half
    min_elev = 90.0 - fov_deg / 2.0  # elevation at outer edge (e.g., 10° if FOV=160°)
    max_elev = 90.0  # center

    # clamp elevation to FOV limits
    elev_deg = max(min_elev, min(max_elev, elev_deg))

    # linear mapping: min_elev -> outer edge (r = half_px), max_elev -> center (r = 0)
    r = (max_elev - elev_deg) / (max_elev - min_elev) * half_px
    return r


@njit
def az_el_to_pixel_fisheye(azimuth, elevation, fv, fc, tilt_n=0.0, tilt_w=0.0, img_size=VIEWER_WINDOW):
    """
    Map azimuth/elevation to pixel coordinates, respecting image bounds, with small tilt compensation.
    tilt_n: degrees, positive -> tilt toward North
    tilt_w: degrees, positive -> tilt toward West
    """
    # Apply tilt compensation
    # North tilt reduces elevation of stars (camera tilted toward N means zenith moves S)
    elevation += tilt_n  
    # West tilt reduces azimuth of stars (camera tilted toward W means zenith moves E)
    azimuth += tilt_w
    
    #r = radius_for_elevation_equisolid(elevation, half, fv, fc)
    r = radius_for_elevation_poly(elevation, half, fv, coeffs)
    angle_rad = np.radians(azimuth)
    x = half + DELTA_X + r * np.sin(angle_rad)
    y = half + DELTA_Y - r * np.cos(angle_rad)
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
def debayer_RGGB2humaneRGB(raw):
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


def get_visible_stars(time_astropy, lat, lon, alt, star_list):
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m) # lat, lon in arcsec
    if debug: print(f"loc: {loc}")
    altaz_frame = AltAz(obstime=time_astropy, location=loc) # coordinate system
    if debug: print(f"altaz_frame: {altaz_frame}")
    visible = []
    for name, ra_h, dec_deg, mag in star_list:
        ra_deg = ra_h * 15.0
        sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='fk5')
        aa = sc.transform_to(altaz_frame)
        az = aa.az.deg
        el = aa.alt.deg
        print(f"name: {name}, ra_h: {ra_h}, ra_deg: {ra_deg}, az: {az}, el: {el}")
        if el > 30.0:  # above horizon
            visible.append((name, az, el, mag))
    return visible


def get_planets(time_astropy, lat, lon, alt, planet_list=None):
    """
    Return a list of planets with (name, ra_deg, dec_deg, az_deg, el_deg, mag_estimate).
    - time_astropy: astropy.time.Time object (UTC)
    - lat, lon in decimal degrees, alt in meters
    - planet_list: optional list of lowercase names, default is the 8 classical planets
    """
    if planet_list is None:
        planet_list = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]  # omit pluto

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    altaz_frame = AltAz(obstime=time_astropy, location=loc)

    planets_out = []

    # use builtin ephemeris (sufficient for many use-cases); change to 'jpl' if installed for higher precision
    with solar_system_ephemeris.set('builtin'):
        for name in planet_list:
            try:
                body = get_body(name, time_astropy, loc)   # returns a SkyCoord in GCRS/ICRS context
            except Exception as e:
                # fallback or skip if body not available
                print(f"get_body({name}) failed: {e}")
                continue

            # RA/Dec in ICRS (degrees)
            try:
                ra_deg = body.icrs.ra.deg
                dec_deg = body.icrs.dec.deg
            except Exception:
                # some astropy versions return different frames — try transform
                icrs = body.transform_to("icrs")
                ra_deg = icrs.ra.deg
                dec_deg = icrs.dec.deg

            # Alt/Az from observer
            aa = body.transform_to(altaz_frame)
            az_deg = aa.az.deg
            el_deg = aa.alt.deg
            
            if el_deg <= 0.0: continue

            planets_out.append((name.capitalize(), ra_deg, dec_deg, az_deg, el_deg, 0.0)) # MAG levels of all planets set to 0.0

    return planets_out


def save_star_points():
    global star_points
    global planet_points
    global expected_star_pixels
    save_path = "../scripts/star_calibration_points.json"
    count = len(expected_star_pixels)
    if count == 0:
        print(f"⚠️  No stars to save → {save_path}")
        return
    data = {
        "fov_deg": FOV,
        "Rrim_px": half,
        "cx_cy": (half, half),
        "tilt_N_W": (TILT_COMP_N, TILT_COMP_W),
        "delta_X_Y": (DELTA_X, DELTA_Y),
        "coeffs": coeffs,
        "stars": star_points,
        "planets": planet_points,
        "expected_star_pixels": expected_star_pixels
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {count} points → {save_path}")
    

def mouse_callback(event, x, y, flags, param):
    global expected_star_pixels, selected_star

    if not stars_view:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        print("LMB hit at", x, y)
        # if a star is currently selected → move it here and unselect
        if selected_star >= 0:
            name = expected_star_pixels[selected_star][0]
            expected_star_pixels[selected_star][1] = x
            expected_star_pixels[selected_star][2] = y
            print(f"Moved {name} to new position: ({x}, {y})")
            selected_star = -1
        else:
            # otherwise, select a star close to the click
            selected_star = -1
            for i, sp in enumerate(expected_star_pixels):
                dx = sp[1] - x
                dy = sp[2] - y
                if dx*dx + dy*dy < 30*30:
                    selected_star = i
                    print(f"Selected star: {sp[0]}")
                    break


def draw_zenith(img, poly, scale=2.0, color=(0,0,128)):
    p = poly.astype(np.float32) * scale
    p[:, 0] += half  # x
    p[:, 1] += half  # y
    pts = p.astype(np.int32)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)

def draw_cardinal(img, poly, target_az_deg, az_offset_deg, scale=2.0, color=(0,0,128)):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(h * 0.48)

    p = poly.astype(np.float32) * scale
    poly_centered = p + np.array([cx, cy], dtype=np.float32)

    global MIRROR_EW
    sign = -1 if not MIRROR_EW else +1
    theta = radians(sign * (az_offset_deg + target_az_deg))

    rot_mat = np.array([[ cos(theta), -sin(theta)],
                        [ sin(theta),  cos(theta)]], dtype=np.float32)

    rotated = (poly_centered - np.array([cx, cy])) @ rot_mat.T + np.array([cx, cy])

    dx = radius * np.sin(theta)
    dy = -radius * np.cos(theta)
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
        [ np.cos(angle_rad), -np.sin(angle_rad)],
        [ np.sin(angle_rad), np.cos(angle_rad)]
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
    global scale_prev, expected_star_pixels, selected_star
    global star_points, visible_stars
    global_time = time.perf_counter()
    img = frame_buffer[index]
    if debug: print(f"Index: {index} ------------------------------------")
    
    ### Image preprocess for viewing
    
    # Debayering - 0.0225s
    start_time = time.perf_counter()
    if humaneRGB:
        debayered_img = debayer_RGGB2humaneRGB(img)
    else:
        debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRGGB2RGB_EA)
    #if debug: print(f"compute debayering: {(time.perf_counter() - start_time):.4f}s")
    
    # Compute per-channel average on downsampled 8-bit for EMA - 
    start_time = time.perf_counter()
    avg_down = debayered_img[::4, ::4, :].mean(axis=(0,1)).astype(np.float32)
    gray = avg_down.mean()
    scale = alpha * (gray / avg_down) + (1 - alpha) * scale_prev
    scale_prev = scale.copy()
    #if debug: print(f"compute EMA scale: {(time.perf_counter() - start_time):.4f}s")
    
    # Update LUT if Gamma or per-channel scale changed
    start_time = time.perf_counter()
    gamma_scaler.update_gamma_scale(gamma=gamma, scale=scale)
    img_8bit = gamma_scaler.apply(debayered_img)
    #if debug: print(f"compute Gamma: {(time.perf_counter() - start_time):.4f}s")

    ### UI
    
    if cardinals_view:
        # Draw zenith point
        draw_zenith(img_8bit, ZENITH_POLY)
        
        # draw outer ring at 30°
        cv2.circle(img_8bit, center, outer_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA) # outer_radius=1600px
        deg_x = int(half - outer_radius + 20)
        cv2.putText(img_8bit, f"30`", (deg_x, int(half)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE * 0.8, (0, 0, 128), 3, cv2.LINE_AA)

        # draw inner ring at 45°
        cv2.circle(img_8bit, center, inner_radius, (0, 0, 128), 2, lineType=cv2.LINE_AA) # inner_radius=800px
        cv2.putText(img_8bit, f"45`", (int(half - inner_radius + 20), int(half)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE * 0.8, (0, 0, 128), 3, cv2.LINE_AA)

        ### --- draw cardinal letters N, E, S, W (respect MIRROR_EW for E/W) ---
        # N: 0 deg, E: 90 deg, S: 180 deg, W: 270 deg
        cardinal_list = [
            (NORTH_POLY, 0.0),
            (E_POLY, 90.0),
            (S_POLY, 180.0),
            (W_POLY, 270.0),
        ]
        for poly, az_deg in cardinal_list:
            # Step 1: raw azimuth (no mirroring, no offset)
            draw_az = az_deg
            
            # Step 2: mirroring
            # If the fisheye is mirrored left-right, swap E <-> W by mirroring their azimuth
            if MIRROR_EW and (az_deg == 90.0 or az_deg == 270.0):
                draw_az = (360.0 - draw_az) % 360.0
            
            # Step 3: apply azimuth offset
            draw_az = (draw_az - AZIMUTH_OFFSET_DEG) % 360.0
            draw_cardinal(img_8bit, poly, draw_az, 0, scale=2.0, color=(0,0,128))

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
        cv2.putText(img_8bit, f"TS: {timestamp_str}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Exposure: {int(metadata_buffer[index][1])} us", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gain: {metadata_buffer[index][2]:.2f} dB", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Gamma: {gamma:.1f}", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"Azimuth offset: {AZIMUTH_OFFSET_DEG:.2f}`", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, f"MAG level: {MAG_LEVEL}", (40, 370), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        if bgs_view: cv2.putText(img_8bit, f"Mask pixels: {num_fg_pixels:.4f} %", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)

        # Debug
        cv2.putText(img_8bit, f"FOV (g/v): {FOV:.2f} deg", (2800, 3140), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE * 0.7, (128, 128, 128), 3, cv2.LINE_AA)

        # Keys
        cv2.putText(img_8bit, "i Info", (40, 2390), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "a ADSB", (40, 2440), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "b BGS", (40, 2490), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "c Celestials", (40, 2540), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "n Cardinals", (40, 2590), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "r humane RGB", (40, 2640), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "+ Gamma ++", (40, 2690), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "- Gamma --", (40, 2740), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "9 Azimuth ++", (40, 2790), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "7 Azimuth --", (40, 2840), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "8 MAG ++", (40, 2890), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "5 MAG --", (40, 2940), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "2 Pause", (40, 2990), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "  3 Forward", (40, 3040), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "  1 Backward", (40, 3090), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        cv2.putText(img_8bit, "q Quit", (40, 3140), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 128), 3, cv2.LINE_AA)
        
        # Flags
        if info_view: cv2.putText(img_8bit, "INFO", (2870,70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if cardinals_view: cv2.putText(img_8bit, "CARDINALS", (2870,120), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if adsb_view: cv2.putText(img_8bit, "ADSB", (2870,170), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if bgs_view: cv2.putText(img_8bit, "BGS", (2870,220), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if stars_view: cv2.putText(img_8bit, "CELESTIALS", (2870,270), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if humaneRGB: cv2.putText(img_8bit, "HUMANE RGB", (2870,320), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), 3, cv2.LINE_AA)
        if paused: cv2.putText(img_8bit, "PAUSED", (2870,370), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 128, 200), 3, cv2.LINE_AA)

    # ADSB ---------------------------------------
    if adsb_view:
        aircraft_screen_coords.clear()

        ### Read aircraft JSON file
        now_ts = time.time()
        try:
            with open(AIRCRAFT_JSON_PATH, 'r') as f:
                aircraft_data = json.load(f)
                
            seen_keys = set()

            for ac in aircraft_data.get("aircraft", []):
                # identify aircraft - prefer unique ICAO24 hex - skipping non icao flights
                icao = ac.get("hex") or ac.get("icao24") or ac.get("icao") or ac.get("squawk") or ac.get("flight","")
                if not icao:
                    continue
                
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

                dist = distance_3d(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, lat, lon, alt) # slanted aka line of sight
                
                # Convert geodetic -> ENU
                x, y, z = geodetic_to_enu(lat, lon, alt, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)

                # Step 0: raw az/elev/heading (no mirroring, no offset)
                az_raw, el_raw = enu_to_az_el(x, y, z)
                heading_raw = mag_heading if mag_heading is not None else 0.0

                # Step 1: mirroring
                if MIRROR_EW:
                    az_after_mirror = (360.0 - az_raw) % 360.0
                    heading_after_mirror = (360.0 - heading_raw) % 360.0
                else:
                    az_after_mirror = az_raw
                    heading_after_mirror = heading_raw
                
                # Step 2: apply azimuth offset
                az_corrected = (az_after_mirror - AZIMUTH_OFFSET_DEG) % 360.0
                heading_corrected = (heading_after_mirror - AZIMUTH_OFFSET_DEG) % 360.0

                px, py = az_el_to_pixel_fisheye(az_corrected, el_raw, fv=FOV, fc=FISHEYE_CURVATURE, tilt_n=TILT_COMP_N, tilt_w=TILT_COMP_W, img_size=VIEWER_WINDOW)

                # Only skip if completely outside image canvas
                if px < 0 or px >= VIEWER_WINDOW or py < 0 or py >= VIEWER_WINDOW:
                    if icao in tracks:
                        tracks[icao]['last_seen'] = now_ts
                    continue

                label = (
                    f"{flight}\n"
                    f"alt:{int(alt)} m\n"
                    f"gs:{int(gs)} km/h\n"
                    f"hdg:{int(mag_heading)} {heading_to_compass(mag_heading)}\n"
                    f"dist:{(dist // 1000):.1f} km\n"
                )

                # Record screen coords for drawing later
                aircraft_screen_coords.append((px, py, label, heading_corrected))

                # Track management
                seen_keys.add(icao)
                t = tracks.get(icao)
                if t is None:
                    dq = deque(maxlen=TRACK_MAX_LEN)
                    dq.append((int(px), int(py)))
                    tracks[icao] = {'points': dq, 'last_seen': now_ts, 'label': label}
                else:
                    last_seen = t['last_seen']
                    pts = t['points']
                    last_px, last_py = pts[-1] if pts else (None, None)
                    # only append if moved a bit to avoid duplicate points
                    if last_px is None or (abs(int(px) - last_px) >= MIN_MOVE_PIXELS or abs(int(py) - last_py) >= MIN_MOVE_PIXELS):
                        pts.append((int(px), int(py)))
                    t['last_seen'] = now_ts
                    t['label'] = label

            # prune stale tracks
            for icao_key, rec in list(tracks.items()):
                if icao_key not in seen_keys and (now_ts - rec['last_seen']) > TRACK_TTL:
                    del tracks[icao_key]

        except Exception as e:
            print("Aircraft JSON read error:", e)

        # Draw tracks (before drawing plane icons so lines are under icons)
        for icao_key, rec in tracks.items():
            pts = rec['points']
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32)
                # optional: choose color per aircraft (hash of key) for easier visual separation
                color_seed = (hash(icao_key) & 0xFFFFFF)
                color = ((color_seed >> 16) & 0xFF, (color_seed >> 8) & 0xFF, color_seed & 0xFF)
                # draw polyline (antialiased). thickness small.
                cv2.polylines(img_8bit, [pts_arr], False, color, thickness=2, lineType=cv2.LINE_AA)

        # Draw aircraft icons (after tracks)
        for px, py, label, mag_heading in aircraft_screen_coords:
            # convert to CCW-positive angle for draw_plane
            heading_raw = mag_heading if mag_heading is not None else 0.0
            heading_draw = (180.0 + heading_raw - AZIMUTH_OFFSET_DEG) % 360.0
            draw_plane(img_8bit, (px, py), heading_draw, scale=0.5)
            #if debug: print(f"heading_raw={heading_raw:.1f}, heading_draw={heading_draw:.1f}")
            draw_text_box(img_8bit, label, (px + 10, py + 20))

    # Stars and planets ---------------------------------------
    if stars_view:
        
        # Draw real star positions
        for name, x, y, mag, az, el in star_points:
            if mag <= MAG_LEVEL:
                cv2.circle(img_8bit, (int(x), int(y)), 8, (0, 255, 0), 2)  # green circles
                cv2.putText(img_8bit, f"{name}, mag={mag}", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
        # Draw real planet positions
        for name, x, y, mag, az, el in planet_points:
            if mag <= MAG_LEVEL:
                cv2.circle(img_8bit, (int(x), int(y)), 8, (0, 128, 255), 2)  # orange circles
                cv2.putText(img_8bit, f"{name}, mag={mag}", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 1)
        
        # Draw expected star positions
        for i, (name, x, y, mag, az, el) in enumerate(expected_star_pixels):
            if mag <= MAG_LEVEL:
                if i == selected_star:
                    color = (255, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.circle(img_8bit, (int(x), int(y)), 10, color, 2)  # blue circles
                cv2.putText(img_8bit, f"{name}, mag={mag}", (int(x)+10, int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

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
cv2.setMouseCallback(window_title, mouse_callback)

center = (VIEWER_WINDOW // 2, VIEWER_WINDOW // 2)

# Outer ring = elevation 30°
outer_radius = int(round(radius_for_elevation_equisolid(30.0, half, FOV, FISHEYE_CURVATURE)))

# Inner ring = elevation 45°
inner_radius = int(round(radius_for_elevation_equisolid(45.0, half, FOV, FISHEYE_CURVATURE)))

while True:
    if paused:
        # Blocking wait for key
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
    
    if key == ord('n'):
        cardinals_view = not cardinals_view
        print(f"Cardinals = {cardinals_view}")
        show_frame(current_index)
        continue

    if key == ord('r'):
        humaneRGB = not humaneRGB
        print(f"humaneRGB = {humaneRGB}")
        show_frame(current_index)
        continue
    
    if key == ord('c'):
        if not stars_view:
            stars_view = True
            print(f"⭐ Star calibration mode\nMove stars to their real positions.")
            
            now = Time(datetime.now(timezone.utc))
            
            # Stars
            visible_stars = get_visible_stars(now, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, BRIGHT_STARS)
            star_points = []
            for name, az, el, mag in visible_stars:
                # Step 1: mirror horizontally (flip East-West)
                az = (360.0 - az) % 360.0
                
                # Step 2: apply camera azimuth offset
                az = (az - AZIMUTH_OFFSET_DEG) % 360.0
                
                # Step 3: project to image coordinates
                x, y = az_el_to_pixel_fisheye(az, el, FOV, FISHEYE_CURVATURE, tilt_n=TILT_COMP_N, tilt_w=TILT_COMP_W, img_size=VIEWER_WINDOW)
                
                # Skip stars outside image bounds
                if not (0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT):
                    print(f"Skipping {name}: out of bounds ({x:.1f}, {y:.1f})")
                    continue
                
                # Add to star list - green circles
                star_points.append((name, x, y, mag, az, el))
                expected_star_pixels.append([name, x, y, mag, -1.0, -1.0])
                if debug: print(f"Star registered: {name} az={az:.2f} el={el:.2f} → x={x:.1f} y={y:.1f} MAG={mag}")
            
            # Planets
            planets = get_planets(now, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)
            planet_points = []
            for name, ra, dec, az, el, mag in planets:
                # Step 1: mirror horizontally (flip East-West)
                az = (360.0 - az) % 360.0
                
                # Step 2: apply camera azimuth offset
                az = (az - AZIMUTH_OFFSET_DEG) % 360.0
                
                # Step 3: project to image coordinates
                x, y = az_el_to_pixel_fisheye(az, el, FOV, FISHEYE_CURVATURE, tilt_n=TILT_COMP_N, tilt_w=TILT_COMP_W, img_size=VIEWER_WINDOW)
                
                # Skip planets outside image bounds
                if not (0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT):
                    print(f"Skipping {name}: out of bounds ({x:.1f}, {y:.1f})")
                    continue
                
                # Add to planet list - orange circles
                planet_points.append((name, x, y, mag, az, el))
                expected_star_pixels.append([name, x, y, mag, -1.0, -1.0])
                if debug: print(f"Planet registered: {name} az={az:.2f} el={el:.2f} → x={x:.1f} y={y:.1f} MAG={mag}")
                
        else:
            if debug:
                print("❌ Star calibration mode DISABLED.")
                print("Refitting star mapping parameters ...")
            refitter = FisheyeRefitter(
                star_points = star_points,
                expected_star_pixels = expected_star_pixels,
                cx_cy = [DELTA_X, DELTA_Y],
                Rrim_px = half,
                fov_deg = FOV
            )
            result = refitter.refit()
            if debug:
                print("Refit result:", result)
                print("Saving new mapping JSON")
            save_star_points()
            stars_view = False
            star_points.clear()
            planet_points.clear()
            expected_star_pixels.clear()

        show_frame(current_index)
        time.sleep(0.01)  # debounce
    
    #######
    if key == ord('g'):
        FOV += 1
        show_frame(current_index)
        continue
    
    if key == ord('v'):
        FOV -= 1
        show_frame(current_index)
        continue
    #######
    
    if key == ord('9'):
        AZIMUTH_OFFSET_DEG += 1
        show_frame(current_index)
        continue
    
    if key == ord('7'):
        AZIMUTH_OFFSET_DEG -= 1
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
    
    if key == ord('8'):
        MAG_LEVEL += MAG_INC
        show_frame(current_index)
        continue
    
    if key == ord('5'):
        MAG_LEVEL -= MAG_INC
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
