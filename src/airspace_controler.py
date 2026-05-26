# \copyright    Sky360.org
#
# \brief        Controller for satellite and aircraft loggers.
#
# ************************************************************************

import sys
import time
from serial.tools import list_ports

from airspace_controler.rtl_adsb_runner import RtlAdsbRunner
from airspace_controler.gps_runner import GpsRunner

from ecal_lib.src.proto_sender import ProtoSender


# ✅ Correct names (match proto)
airplane_channel_name = "airplane log"
airplane_message_name = "AdsbMessage"
airplane_proto_file = "ecal_lib.proto_files.adsblog_pb2"

satellite_channel_name = "satellite log"
satellite_message_name = "GpsLog"
satellite_proto_file = "ecal_lib.proto_files.gpslog_pb2"


# ---------- ADS-B ----------
def populate_airplane_message(payload, airplane):
    # payload is "repeated Adsb_data"
    payload.clear()

    msg = payload.add()

    msg.raw_line = airplane.get("raw_line", "")
    msg.hex = airplane.get("hex", "")
    msg.df = int(airplane.get("DF", 0))
    msg.icao = airplane.get("ICAO", "")
    msg.type_code = int(airplane.get("TypeCode", 0))

    msg.altitude_m = float(airplane.get("Altitude_m", 0.0))
    msg.latitude = float(airplane.get("Latitude", 0.0))
    msg.longitude = float(airplane.get("Longitude", 0.0))

    msg.azimuth_deg = float(airplane.get("Azimuth_deg", 0.0))
    msg.elevation_deg = float(airplane.get("Elevation_deg", 0.0))
    msg.range_m = float(airplane.get("Range_m", 0.0))

    # ✅ add timestamps (important for your system consistency)
    now_ns = time.time_ns()
    msg.epoch_time.sec = now_ns // 1_000_000_000
    msg.epoch_time.nsec = now_ns % 1_000_000_000

    mono_ns = time.monotonic_ns()
    msg.mono_time.sec = mono_ns // 1_000_000_000
    msg.mono_time.nsec = mono_ns % 1_000_000_000


# ---------- GPS ----------
def populate_satellite_message(payload, satellite):
    # payload is GpsData (contains oneof)
    payload.ClearField("log_type")

    now_ns = time.time_ns()
    mono_ns = time.monotonic_ns()

    if satellite["type"] == "Global Positioning":
        gp = payload.global_positioning

        gp.utc_time = satellite.get("utc_time", "")
        gp.latitude = satellite.get("latitude", 0.0)
        gp.lat_dir = satellite.get("lat_dir", "")
        gp.longitude = satellite.get("longitude", 0.0)
        gp.lon_dir = satellite.get("lon_dir", "")

        gp.speed_knots = satellite.get("speed_knots", 0.0)
        gp.speed_kmh = satellite.get("speed_kmh", 0.0)
        gp.course = satellite.get("course", 0.0) or 0.0

        gp.date = satellite.get("date", "")
        gp.status = satellite.get("status", "")

        gp.fix_quality = int(satellite.get("fix_quality", 0))
        gp.satellites_used = int(satellite.get("satellites_used", 0))
        gp.hdop = float(satellite.get("hdop", 0.0) or 0.0)

        gp.altitude = float(satellite.get("altitude", 0.0))
        gp.alt_units = satellite.get("alt_units", "")

        gp.easting = float(satellite.get("easting", 0.0))
        gp.northing = float(satellite.get("northing", 0.0))

        gp.zone = int(satellite.get("zone", 0))
        gp.northern = bool(satellite.get("northern", True))

        gp.epoch_time.sec = now_ns // 1_000_000_000
        gp.epoch_time.nsec = now_ns % 1_000_000_000
        gp.mono_time.sec = mono_ns // 1_000_000_000
        gp.mono_time.nsec = mono_ns % 1_000_000_000

    elif satellite["type"] == "GNSS DOP and Active Satellites":
        dop = payload.dop_active

        dop.mode = satellite.get("mode", "")
        dop.fix_type = satellite.get("fix_type", "")
        dop.satellites_in_use.extend(satellite.get("satellites_in_use", []))

        dop.pdop = float(satellite.get("pdop", 0.0) or 0.0)
        dop.hdop = float(satellite.get("hdop", 0.0) or 0.0)
        dop.vdop = float(satellite.get("vdop", 0.0) or 0.0)

        dop.epoch_time.sec = now_ns // 1_000_000_000
        dop.epoch_time.nsec = now_ns % 1_000_000_000
        dop.mono_time.sec = mono_ns // 1_000_000_000
        dop.mono_time.nsec = mono_ns % 1_000_000_000

    elif satellite["type"] == "Satellites in View":
        siv = payload.satellites_in_view

        siv.talker = satellite.get("talker", "")
        siv.total_in_view = int(satellite.get("total_in_view", 0))

        for sat in satellite.get("satellites", []):
            sat_msg = siv.satellites.add()

            sat_msg.prn = sat.get("prn", "")
            sat_msg.elevation_deg = float(sat.get("elevation_deg", 0.0) or 0.0)
            sat_msg.azimuth_deg = float(sat.get("azimuth_deg", 0.0) or 0.0)

            if sat.get("sat_unit_vec"):
                sat_msg.sat_unit_vec.extend(sat["sat_unit_vec"])

            sat_msg.snr = float(sat.get("snr", 0.0) or 0.0)

        siv.epoch_time.sec = now_ns // 1_000_000_000
        siv.epoch_time.nsec = now_ns % 1_000_000_000
        siv.mono_time.sec = mono_ns // 1_000_000_000
        siv.mono_time.nsec = mono_ns % 1_000_000_000


# ---------- MAIN ----------
if __name__ == "__main__":

    port_list = list(list_ports.comports())

    gps_port = "COM3" if sys.platform.startswith("win") else "/dev/ttyUSB0"

    if port_list:
        gps_port = port_list[0].device

    print(f"Using GPS port: {gps_port}")

    satellite_detector = GpsRunner(port=gps_port, baud_rate=9600)
    satellite_detector.run()

    my_location = satellite_detector.get_my_location()
    print(f"GPS location = {my_location}")

    airplane_detector = RtlAdsbRunner()
    airplane_detector.set_my_position(*my_location)
    airplane_detector.run()

    airplane_proto_snd = ProtoSender(
        airplane_channel_name,
        airplane_message_name,
        airplane_proto_file
    )

    satellite_proto_snd = ProtoSender(
        satellite_channel_name,
        satellite_message_name,
        satellite_proto_file
    )

    try:
        while True:

            # ---------- ADS-B ----------
            airplane_logs = airplane_detector.get_detection()
            if airplane_logs:
                payload = airplane_proto_snd.get_message_type()
                populate_airplane_message(payload, airplane_logs[0])
                airplane_proto_snd.send(payload)

            # ---------- GPS ----------
            satellite_logs = satellite_detector.get_log()
            if satellite_logs:
                payload = satellite_proto_snd.get_message_type()
                populate_satellite_message(payload, satellite_logs[0])
                satellite_proto_snd.send(payload)

    except KeyboardInterrupt:
        airplane_detector.close()
        satellite_detector.close()