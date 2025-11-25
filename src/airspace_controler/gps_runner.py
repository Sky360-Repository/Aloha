# \copyright    Sky360.org
#
# \brief        GPS runner - interface to GPS module.
#
# ************************************************************************

import serial
from serial.tools import list_ports
import pynmea2
import threading
from .utils import gps_to_utm, sat_to_unit_vector


class GpsRunner:
    def __init__(self, port='COM3', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self._logs = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Per-talker GSV buffers
        self._gsv_state = {}

        # Unified Global Positioning buffer
        self._gp_state = {}           # accumulated fields
        self._gp_epoch_ts = None      # current epoch timestamp (str, e.g. '16:29:57+00:00')
        self._gp_last_emit_ts = None  # last timestamp we emitted a unified snapshot

        self.my_position = None

    def _reader_loop(self):
        while self._running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line.startswith('$'):
                    continue
                msg = pynmea2.parse(line)
                log_entry = self.format_sentence(msg)
                if log_entry is not None:
                    with self._lock:
                        self._logs.append(log_entry)
            except pynmea2.ParseError:
                continue
            except Exception:
                continue

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            print(f"GPS port {self.port} opened at {self.baud_rate} baud.")
        except serial.SerialException as e:
            print(f"Could not open port {self.port}: {e}")
            print("Available ports:")
            for p in list_ports.comports():
                print(f" - {p.device} ({p.description})")

    def get_log(self):
        with self._lock:
            if not self._logs:
                return None
            logs = self._logs[:]
            self._logs.clear()
        return logs

    def get_my_location(self):
        while self.my_position is None:
            self.get_log()
        return [self.my_position["latitude"], self.my_position["longitude"], self.my_position["altitude"]]

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("GPS port closed.")

    # --- Helpers ---
    @staticmethod
    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def kmh_from_knots(knots):
        f = GpsRunner.as_float(knots)
        return None if f is None else f * 1.852

    def _maybe_emit_gp(self, force=False):
        if "latitude" not in self._gp_state or "longitude" not in self._gp_state:
            return None

        current_ts = self._gp_epoch_ts
        if not current_ts:
            return None

        # Only emit on force (GGA), skip otherwise
        if not force:
            return None

        # Prevent duplicate emission in the same second
        if self._gp_last_emit_ts == current_ts:
            return None

        dic = {"type": "Global Positioning"}
        dic.update(self._gp_state)

        self.my_position = dic.copy()
        self._gp_last_emit_ts = current_ts
        return dic

    def format_sentence(self, msg):
        # --- Global Positioning (unified) ---
        if isinstance(msg, pynmea2.GGA):
            lat, lon = msg.latitude, msg.longitude
            easting, northing, zone, northern = gps_to_utm(lat, lon)

            # Update epoch timestamp from GGA
            self._gp_epoch_ts = str(msg.timestamp) if getattr(msg, "timestamp", None) else self._gp_epoch_ts

            # Accumulate fields
            self._gp_state.update({
                "utc_time": str(msg.timestamp),
                "latitude": lat,
                "lat_dir": msg.lat_dir,
                "longitude": lon,
                "lon_dir": msg.lon_dir,
                "fix_quality": msg.gps_qual,
                "satellites_used": msg.num_sats,
                "hdop": getattr(msg, "hdop", None),
                "altitude": msg.altitude,
                "alt_units": msg.altitude_units,
                "easting": easting,
                "northing": northing,
                "zone": zone,
                "northern": northern,
            })

            # Emit unified snapshot (force on GGA)
            return self._maybe_emit_gp(force=True)

        elif isinstance(msg, pynmea2.RMC):
            speed_kn = self.as_float(msg.spd_over_grnd)
            speed_km = self.kmh_from_knots(msg.spd_over_grnd)

            # Update epoch timestamp from RMC
            self._gp_epoch_ts = str(msg.timestamp) if getattr(msg, "timestamp", None) else self._gp_epoch_ts

            # Accumulate fields (RMC carries lat/lon/time/speed/course)
            self._gp_state.update({
                "utc_time": str(msg.timestamp),
                "latitude": msg.latitude,
                "lat_dir": msg.lat_dir,
                "longitude": msg.longitude,
                "lon_dir": msg.lon_dir,
                "speed_knots": speed_kn,
                "speed_kmh": speed_km,
                "course": msg.true_course,
                "date": str(msg.datestamp),
                "status": msg.status,
            })

            # Emit unified snapshot (not forced; avoid duplicate in same second)
            return self._maybe_emit_gp(force=False)

        elif isinstance(msg, pynmea2.VTG):
            kn = self.as_float(getattr(msg, "spd_over_grnd_knots", None))
            km = self.as_float(getattr(msg, "spd_over_grnd_kmph", None))

            # VTG does not carry time; just accumulate speed fields
            self._gp_state.update({
                "speed_knots": kn if kn is not None else self._gp_state.get("speed_knots"),
                "speed_kmh": km if km is not None else self._gp_state.get("speed_kmh"),
            })

            # Emit unified snapshot only if we already have a timestamp and lat/lon
            return self._maybe_emit_gp(force=False)

        # --- GNSS DOP and Active Satellites (GSA) ---
        elif isinstance(msg, pynmea2.GSA):
            svs = []
            for i in range(1, 13):
                field = getattr(msg, f"sv_id{i:02}", None)
                if field:
                    svs.append(field)
            return {
                "type": "GNSS DOP and Active Satellites",
                "mode": msg.mode,
                "fix_type": msg.mode_fix_type,  # 1=no fix, 2=2D, 3=3D
                "satellites_in_use": svs,
                "pdop": getattr(msg, "pdop", None),
                "hdop": getattr(msg, "hdop", None),
                "vdop": getattr(msg, "vdop", None),
            }

        # --- Geographic Position (GLL) ---
        elif isinstance(msg, pynmea2.GLL):
            # Some receivers emit GLL regularly; we can fold it in but do not force emit
            self._gp_epoch_ts = str(getattr(msg, "timestamp", "")) if getattr(msg, "timestamp", None) else self._gp_epoch_ts
            self._gp_state.update({
                "utc_time": str(getattr(msg, "timestamp", "")),
                "latitude": msg.latitude,
                "lat_dir": msg.lat_dir,
                "longitude": msg.longitude,
                "lon_dir": msg.lon_dir,
                "status": msg.status,  # A=valid, V=void
            })
            return self._maybe_emit_gp(force=False)

        # --- Satellites in View (GSV), with per-talker buffering ---
        elif isinstance(msg, pynmea2.GSV):
            talker = getattr(msg, "talker", None) or str(msg).split(',')[0][1:3]
            total = getattr(msg, "num_sv_in_view", None)
            num_messages = int(getattr(msg, "num_messages", 1))
            msg_num = int(getattr(msg, "msg_num", 1))

            sats = []
            for i in range(4):
                prn = getattr(msg, f"sv_prn_num_{i + 1}", None)
                elev = self.safe_float(getattr(msg, f"elevation_deg_{i + 1}", None))
                azim = self.safe_float(getattr(msg, f"azimuth_{i + 1}", None))
                snr = self.safe_float(getattr(msg, f"snr_{i + 1}", None))
                if prn:
                    vec = sat_to_unit_vector(elev, azim)
                    sats.append({
                        "prn": prn,
                        "elevation_deg": elev,
                        "azimuth_deg": azim,
                        "sat_unit_vec": vec.tolist() if hasattr(vec, "tolist") else vec,
                        "snr": snr,
                    })

            st = self._gsv_state.get(talker)
            if msg_num == 1 or st is None:
                st = {"expected": num_messages, "buffer": [], "last_total": total}
                self._gsv_state[talker] = st
            else:
                if st["expected"] != num_messages:
                    st = {"expected": num_messages, "buffer": [], "last_total": total}
                    self._gsv_state[talker] = st

            st["buffer"].extend(sats)
            st["last_total"] = total

            if msg_num == st["expected"]:
                dic = {
                    "type": "Satellites in View",
                    "talker": talker,
                    "total_in_view": st["last_total"],
                    "satellites": st["buffer"],
                }
                self._gsv_state[talker] = {"expected": 0, "buffer": [], "last_total": total}
                return dic
            else:
                return None

        else:
            return {"type": "Unhandled", "raw": str(msg)}


if __name__ == "__main__":
    # Windows: COM3, COM4, etc.
    # Linux/Ubuntu: /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyS0, etc. (depending on whether it’s a USB‑serial adapter or a built‑in port).
    gps_runner = GpsRunner(port='COM3', baud_rate=9600)
    gps_runner.run()

    try:
        while True:
            gps_log = gps_runner.get_log()
            if gps_log:
                for log in gps_log:
                    print(log)
    except KeyboardInterrupt:
        gps_runner.close()