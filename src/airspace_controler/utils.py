# \copyright    Sky360.org
#
# \brief        Support functions for airspace controler
#
# ************************************************************************

import math
import numpy as np
from pyproj import Transformer
import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def gps_to_utm(lat, lon):
    zone = int(math.floor((lon + 180) / 6) + 1)
    northern = lat >= 0
    epsg_code = f"326{zone:02d}" if northern else f"327{zone:02d}"
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone, northern


def utm_to_gps(easting, northing, zone, northern):
    epsg_code = f"326{zone:02d}" if northern else f"327{zone:02d}"
    transformer = Transformer.from_crs(f"epsg:{epsg_code}", "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def sat_to_unit_vector(elev_deg, az_deg) -> np.ndarray:
    if elev_deg is not None and az_deg is not None:
        el = math.radians(float(elev_deg))
        az = math.radians(float(az_deg))
        x = math.cos(el) * math.sin(az)   # East
        y = math.cos(el) * math.cos(az)   # North
        z = math.sin(el)                  # Up
        vec = np.array([x, y, z])
        return vec / np.linalg.norm(vec)  # normalize
    return np.zeros(3)

