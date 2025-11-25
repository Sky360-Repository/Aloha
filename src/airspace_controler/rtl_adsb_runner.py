# \copyright    Sky360.org
#
# \brief        Interface to RTL-SDR to log ADS-B signals.
#
# ************************************************************************

import subprocess
from datetime import datetime, timezone
import re
import math
import pyModeS as pms
from collections import defaultdict
from .utils import gps_to_utm
import threading


class CPRBuffer:
    def __init__(self):
        self.buf = defaultdict(lambda: {"even": None, "odd": None})

    def add(self, icao: str, msg: str, ts: float):
        try:
            if not pms.adsb.is_airborne_position(msg):
                return
            odd_flag = pms.adsb.odd_flag(msg)
            key = "odd" if odd_flag else "even"
            self.buf[icao][key] = (msg, ts)
        except Exception:
            pass

    def decode_if_ready(self, icao: str):
        entry = self.buf.get(icao)
        if not entry or not entry["even"] or not entry["odd"]:
            return None
        meven, _ = entry["even"]
        modd, _ = entry["odd"]
        try:
            lat, lon = pms.adsb.position(meven, modd)
            self.buf[icao] = {"even": None, "odd": None}
            return lat, lon
        except Exception:
            return None


class RtlAdsbRunner:
    def __init__(self, cmd=["rtl_adsb", "-V", "-p", "30003"]):
        self.RTL_ADSB_CMD = cmd
        self.proc = None
        self.cpr = CPRBuffer()
        self.my_gps_position = None
        self.my_position_m = None
        self._detections = []
        self._lock = threading.Lock()
        self._running = False

    def set_my_position(self, lat, lon, alt):
        self.my_gps_position = [lat, lon]
        easting, northing, zone, northern = gps_to_utm(lat, lon)
        self.my_position_m = [easting, northing, alt]

    def relative_position(self, plane_east, plane_north, plane_alt):
        dx = plane_east - self.my_position_m[0]
        dy = plane_north - self.my_position_m[1]
        dz = plane_alt - self.my_position_m[2]
        range_m = math.sqrt(dx**2 + dy**2 + dz**2)
        azimuth_deg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        elevation_deg = math.degrees(math.atan2(dz, math.sqrt(dx**2 + dy**2)))
        return azimuth_deg, elevation_deg, range_m

    def parse_rtl_adsb_output(self, lines):
        msg = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("*") and line.endswith(";"):
                msg = {"raw_line": line, "hex": line[1:-1]}
            elif line.startswith("DF=") or "ICAO Address=" in line:
                df_match = re.search(r"DF=(\d+)", line)
                icao_match = re.search(r"ICAO Address=([0-9a-fA-F]+)", line)
                pi_match = re.search(r"PI=0x([0-9a-fA-F]+)", line)
                type_match = re.search(r"Type Code=(\d+)", line)
                s_match = re.search(r"S\.Type/Ant\.=(\d+)", line)

                if df_match: msg["DF"] = int(df_match.group(1))
                if icao_match: msg["ICAO"] = icao_match.group(1)
                if pi_match: msg["PI"] = pi_match.group(1)
                if type_match: msg["TypeCode"] = int(type_match.group(1))
                if s_match: msg["S_Type"] = int(s_match.group(1))

                if "hex" in msg and "ICAO" in msg:
                    hexmsg = msg["hex"]
                    df = msg.get("DF")
                    try:
                        if df in (17, 18):
                            tc = pms.adsb.typecode(hexmsg)
                            msg["TypeCode"] = tc
                            if 9 <= tc <= 18:
                                alt = pms.adsb.altitude(hexmsg)
                                msg["Altitude_m"] = alt * 0.3048 if alt else None
                                self.cpr.add(msg["ICAO"], hexmsg, datetime.now().timestamp())
                                pos = self.cpr.decode_if_ready(msg["ICAO"])
                                if pos:
                                    msg["Latitude"], msg["Longitude"] = pos
                                else:
                                    try:
                                        lat, lon = pms.adsb.position_with_ref(
                                            hexmsg,
                                            self.my_gps_position[0],
                                            self.my_gps_position[1]
                                        )
                                        msg["Latitude"], msg["Longitude"] = lat, lon
                                        easting, northing, _, _ = gps_to_utm(lat, lon)
                                        msg["Azimuth_deg"], msg["Elevation_deg"], msg["Range_m"] = \
                                            self.relative_position(easting, northing, msg["Altitude_m"])
                                    except Exception:
                                        pass
                            else:
                                msg["Altitude_m"] = None
                        elif df == 19:
                            msg["Altitude_m"] = None
                        else:
                            msg["Altitude_m"] = None
                    except Exception:
                        msg["Altitude_m"] = None

                if msg.get("ICAO") and msg.get("DF") is not None:
                    yield msg

    def run(self):
        self.proc = subprocess.Popen(self.RTL_ADSB_CMD,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     text=True)
        self._running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        for msg in self.parse_rtl_adsb_output(self.proc.stdout):
            if not self._running:
                break
            ts = datetime.now(timezone.utc).isoformat()
            if not any([msg.get("Altitude_m"), msg.get("Latitude"), msg.get("Longitude")]):
                continue
            msg["timestamp"] = ts
            with self._lock:
                self._detections.append(msg)

    def get_detection(self):
        with self._lock:
            if not self._detections:
                return None
            detections = self._detections[:]
            self._detections.clear()
        return detections

    def close(self):
        self._running = False
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


if __name__ == "__main__":
    airplane_detector = RtlAdsbRunner()
    airplane_detector.set_my_position(37.8436571, -116.7303739, 1680)
    airplane_detector.run()

    try:
        while True:
            detections = airplane_detector.get_detection()
            if detections:
                for airplane in detections:
                    print(airplane)
    except KeyboardInterrupt:
        airplane_detector.close()