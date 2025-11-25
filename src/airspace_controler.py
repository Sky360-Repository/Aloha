# \copyright    Sky360.org
#
# \brief        Controller for satellite and aircraft loggers.
#
# ************************************************************************

from airspace_controler.rtl_adsb_runner import RtlAdsbRunner
from airspace_controler.gps_runner import GpsRunner

from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.set_process_name import *


airplane_channel_name = "airplane log"
airplane_message_name = "AdsbLog"
airplane_proto_file = "ecal_lib.proto_files.adsblog_pb2"

satellite_channel_name = "satellite log"
satellite_message_name = "GpsLog"
satellite_proto_file = "ecal_lib.proto_files.gpslog_pb2"

def populate_airplane_message(airplane_message, airplane):
    # Clear previous oneof
    airplane_message.Clear()

    # Populate the message
    msg = airplane_message.messages.add()
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
    msg.timestamp = airplane.get("timestamp", "")

    return airplane_message


def populate_satellite_message(satellite_message, satellite):

    # Clear previous oneof
    satellite_message.Clear()

    if satellite["type"] == "Global Positioning":
        gp = satellite_message.global_positioning
        gp.utc_time = satellite.get("utc_time", "")
        gp.latitude = satellite.get("latitude", 0.0)
        gp.lat_dir = satellite.get("lat_dir", "")
        gp.longitude = satellite.get("longitude", 0.0)
        gp.lon_dir = satellite.get("lon_dir", "")
        gp.speed_knots = satellite.get("speed_knots", 0.0)
        gp.speed_kmh = satellite.get("speed_kmh", 0.0)
        gp.course = satellite.get("course", 0.0) if satellite.get("course") else 0.0
        gp.date = satellite.get("date", "")
        gp.status = satellite.get("status", "")
        gp.fix_quality = int(satellite.get("fix_quality", 0))
        gp.satellites_used = int(satellite.get("satellites_used", 0))
        gp.hdop = float(satellite.get("hdop", 0.0)) if satellite.get("hdop") else 0.0
        gp.altitude = float(satellite.get("altitude", 0.0))
        gp.alt_units = satellite.get("alt_units", "")
        gp.easting = float(satellite.get("easting", 0.0))
        gp.northing = float(satellite.get("northing", 0.0))
        gp.zone = int(satellite.get("zone", 0))
        gp.northern = bool(satellite.get("northern", True))

    elif satellite["type"] == "GNSS DOP and Active Satellites":
        dop = satellite_message.dop_active
        dop.mode = satellite.get("mode", "")
        dop.fix_type = satellite.get("fix_type", "")
        dop.satellites_in_use.extend(satellite.get("satellites_in_use", []))
        dop.pdop = float(satellite.get("pdop", 0.0)) if satellite.get("pdop") else 0.0
        dop.hdop = float(satellite.get("hdop", 0.0)) if satellite.get("hdop") else 0.0
        dop.vdop = float(satellite.get("vdop", 0.0)) if satellite.get("vdop") else 0.0

    elif satellite["type"] == "Satellites in View":
        siv = satellite_message.satellites_in_view
        siv.talker = satellite.get("talker", "")
        siv.total_in_view = int(satellite.get("total_in_view", 0))
        for sat in satellite.get("satellites", []):
            sat_msg = siv.satellites.add()
            sat_msg.prn = sat.get("prn", "")
            sat_msg.elevation_deg = float(sat.get("elevation_deg", 0.0)) if sat.get("elevation_deg") else 0.0
            sat_msg.azimuth_deg = float(sat.get("azimuth_deg", 0.0)) if sat.get("azimuth_deg") else 0.0
            if "sat_unit_vec" in sat and sat["sat_unit_vec"]:
                sat_msg.sat_unit_vec.extend(sat["sat_unit_vec"])
            sat_msg.snr = float(sat.get("snr", 0.0)) if sat.get("snr") else 0.0

    return satellite_message

if __name__ == "__main__":

    # Windows: COM3, COM4, etc.
    # Linux/Ubuntu: /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyS0, etc. (depending on whether it’s a USB-serial adapter or a built-in port).
    satellite_detector = GpsRunner(port='COM3', baud_rate=9600)
    satellite_detector.run()

    my_location = satellite_detector.get_my_location()
    print(f"GPS location = {my_location}")

    airplane_detector = RtlAdsbRunner()
    airplane_detector.set_my_position(my_location[0], my_location[1], my_location[2])
    airplane_detector.run()

    # Proto sender for airplane log and satellite log
    set_process_name(f"airspace controler Broadcast")
    airplane_proto_snd = ProtoSender(airplane_channel_name, airplane_message_name, airplane_proto_file)
    satellite_proto_snd = ProtoSender(satellite_channel_name, satellite_message_name, satellite_proto_file)

    try:
        while True:
            airplane_logs = airplane_detector.get_detection()
            if airplane_logs:
                for airplane in airplane_logs:
                    # Access the protobuf type definition
                    airplane_message = airplane_proto_snd.get_message_type()

                    # Populate the message
                    airplane_message = populate_airplane_message(airplane_message, airplane)

                    # Send the message to the topic this publisher was created for
                    airplane_proto_snd.send(airplane_message)

            satellite_log = satellite_detector.get_log()
            if satellite_log:
                for satellite in satellite_log:
                    # Access the protobuf type definition
                    satellite_message = satellite_proto_snd.get_message_type()

                    # Populate the message
                    satellite_message = populate_satellite_message(satellite_message, satellite)

                    # Send the message to the topic this publisher was created for
                    satellite_proto_snd.send(satellite_message)

    except KeyboardInterrupt:
        airplane_detector.close()
        satellite_detector.close()