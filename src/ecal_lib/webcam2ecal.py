# \copyright    Sky360.org
#
# \brief        Captures video from a webcam and  broadcasts over eCAL.
#
# ************************************************************************

import time
import cv2
import argparse

from ecal_lib.src.proto_sender import ProtoSender
from ecal_lib.utils.jpeg_compression import convert_image_to_byte_array


def _fill_timestamp(ts_msg, ns_value: int):
    """
    Fill a protobuf Timestamp message from a nanosecond integer.
    """
    ts_msg.sec = ns_value // 1_000_000_000
    ts_msg.nsec = ns_value % 1_000_000_000


def webcam2ecal(camera_id, channel_name, wrapper_message_name, proto_file):
    # JPEG compression rate in [0, 100]
    # where 100 is best quality.
    # To record raw only set jpeg_quality = -1
    jpeg_quality = 100

    # Open webcam
    cap = cv2.VideoCapture(camera_id)

    cv2.namedWindow(channel_name + " Broadcast", cv2.WINDOW_NORMAL)

    # Try to force a useful webcam setup
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print(f"max_width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"max_height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

    # Proto sender
    # IMPORTANT:
    # wrapper_message_name must be the wrapper type, i.e. "web_camera_image"
    # ProtoSender.get_message_type() will return the payload type, i.e. frame_data
    proto_snd = ProtoSender(channel_name, wrapper_message_name, proto_file)

    corr_time = time.time_ns() // 1000
    avg_run_time = 1.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Camera {camera_id} not available")
            break

        # Payload message (this is frame_data, not the wrapper)
        protobuf_message = proto_snd.get_message_type()

        # Frame shape
        protobuf_message.width = frame.shape[1]
        protobuf_message.height = frame.shape[0]

        # OpenCV BGR image is usually 8 bits per channel
        # If you want "total bits per pixel", 3 * 8 = 24
        protobuf_message.bit_per_pixel = 24

        if jpeg_quality == -1:
            protobuf_message.raw_image = frame.tobytes()
            protobuf_message.jpeg_data = b""
            protobuf_message.jpeg_size = 0
        else:
            byte_array = convert_image_to_byte_array(frame, jpeg_quality)
            protobuf_message.jpeg_data = byte_array
            protobuf_message.jpeg_size = len(byte_array)
            protobuf_message.raw_image = b""

        protobuf_message.sensor_id = int(camera_id)

        # Acquisition timestamps stored in payload
        epoch_ns = time.time_ns()
        mono_ns = time.monotonic_ns()

        _fill_timestamp(protobuf_message.epoch_time, epoch_ns)
        _fill_timestamp(protobuf_message.mono_time, mono_ns)

        # Metadata
        protobuf_message.source = "webcam2ecal"
        protobuf_message.frame_id = channel_name

        # Send payload only — ProtoSender wraps it and adds Header automatically
        proto_snd.send(protobuf_message)

        # Runtime frequency overlay
        prev_time = corr_time
        corr_time = time.time_ns() // 1000
        avg_run_time = 0.9 * avg_run_time + 0.1 * (corr_time - prev_time)
        run_freq = 1.0e6 / avg_run_time if avg_run_time > 0 else 0.0

        frame_overlay = cv2.putText(
            frame.copy(),
            f"run_freq: {run_freq:.2f} Hz",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        frame_overlay = cv2.putText(
            frame_overlay,
            f"Image Size: [{frame.shape[1]}, {frame.shape[0]}]",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(channel_name + " Broadcast", frame_overlay)

        # ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel_name", required=False,
                    help="Specify the channel name: python -m ecal_lib.webcam2ecal --channel_name camera_0")
    ap.add_argument("--camera_id", required=False,
                    help="Specify Camera ID: python -m ecal_lib.webcam2ecal --camera_id 0")
    ap.add_argument("--message_name", required=False,
                    help="Specify the WRAPPER message name: python -m ecal_lib.webcam2ecal --message_name web_camera_image")
    ap.add_argument("--proto_file", required=False,
                    help="Specify proto_file: python -m ecal_lib.webcam2ecal --proto_file ecal_lib.proto_files.web_camera_image_pb2")
    args = vars(ap.parse_args())

    # Default config
    camera_id = 0
    channel_name = "camera_" + str(camera_id)

    # IMPORTANT:
    # This must be the wrapper type, not frame_data
    wrapper_message_name = "web_camera_image"

    # Use package-style import path
    proto_file = "ecal_lib.proto_files.web_camera_image_pb2"

    print("\nDefault usage:")
    print("python -m ecal_lib.webcam2ecal "
          "--camera_id 0 "
          "--channel_name camera_0 "
          "--message_name web_camera_image "
          "--proto_file ecal_lib.proto_files.web_camera_image_pb2")
    print("\nPress 'Esc' key to stop\n")

    if args["channel_name"]:
        channel_name = args["channel_name"]
    if args["camera_id"]:
        camera_id = int(args["camera_id"])
    if args["message_name"]:
        wrapper_message_name = args["message_name"]
    if args["proto_file"]:
        proto_file = args["proto_file"]

    webcam2ecal(camera_id, channel_name, wrapper_message_name, proto_file)
