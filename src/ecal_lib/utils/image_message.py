# \copyright    Sky360.org
#
# \brief        Strategy design to handle Image massages.
#
# ************************************************************************

import numpy as np
import cv2
from abc import ABC, abstractmethod

from .jpeg_compression import convert_byte_array_to_image


# ---------- Strategy interface ----------
class ImageStrategy(ABC):
    @abstractmethod
    def process_message(self, message):
        pass

    @abstractmethod
    def get_rgb_image(self):
        pass

    @abstractmethod
    def get_time_stamp(self):
        pass


# ---------- Strategy for new generic frame_data ----------
class WebCameraImageStrategy(ImageStrategy):
    def __init__(self):
        self.time_stamp = 0
        self.rgb = None

    def process_message(self, message):
        if message.jpeg_size == 0:
            frame = np.frombuffer(message.raw_image, dtype=np.uint8)
            self.rgb = frame.reshape((message.height, message.width, 3))
        else:
            self.rgb = convert_byte_array_to_image(message.jpeg_data)

        # Use epoch_time from new schema
        u_sec = np.uint64(message.epoch_time.sec)
        u_nano_sec = np.uint64(message.epoch_time.nsec)
        self.time_stamp = np.uint64((u_sec * 1_000_000_000) + u_nano_sec)

    def get_rgb_image(self):
        return self.rgb

    def get_time_stamp(self):
        return self.time_stamp

# ---------- Legacy QHY strategy ----------
class QHYImageStrategy(ImageStrategy):
    def __init__(self):
        self.time_stamp = 0
        self.rgb = None

    def process_message(self, message):
        frame = np.frombuffer(message.raw_image, dtype=np.uint16)
        frame = frame.reshape((message.height, message.width, 1))
        debayered_img = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
        self.rgb = debayered_img[::4, ::4]

        if hasattr(message, "epoch_time"):
            u_sec = np.uint64(message.epoch_time.sec)
            u_nano_sec = np.uint64(message.epoch_time.nsec)
            self.time_stamp = np.uint64((u_sec * 1_000_000_000) + u_nano_sec)
        elif hasattr(message, "time_stamp"):
            self.time_stamp = message.time_stamp
        else:
            self.time_stamp = 0

    def get_rgb_image(self):
        return self.rgb

    def get_time_stamp(self):
        return self.time_stamp


# ---------- Legacy CameraImage strategy ----------
class CameraImageStrategy(ImageStrategy):
    def __init__(self):
        self.time_stamp = 0
        self.rgb = None

    def process_message(self, message):
        if message.jpeg_size == 0:
            frame = np.frombuffer(message.raw_image, dtype=np.uint8)
            self.rgb = frame.reshape((message.height, message.width, 3))
        else:
            self.rgb = convert_byte_array_to_image(message.jpeg_data)

        # legacy t_Stamp field
        u_sec = np.uint64(message.t_Stamp.u_sec)
        u_nano_sec = np.uint64(message.t_Stamp.u_nano_sec)
        self.time_stamp = np.uint64((u_sec * 1_000_000_000) + u_nano_sec)

    def get_rgb_image(self):
        return self.rgb

    def get_time_stamp(self):
        return self.time_stamp


# ---------- Context ----------
class ImageMessage:
    def __init__(self, message: str):
        if message == "web_camera_image":
            self._strategy = WebCameraImageStrategy()

        elif message == "qhy_camera_image":
            self._strategy = QHYImageStrategy()

        elif message == "CameraImage":
            self._strategy = CameraImageStrategy()

        else:
            raise ValueError(
                f"Invalid message type '{message}'. "
                f"Expected one of: web_camera_image, qhy_camera_image, CameraImage"
            )

    def process_message(self, message):
        return self._strategy.process_message(message)

    def get_rgb_image(self):
        return self._strategy.get_rgb_image()

    def get_time_stamp(self):
        return self._strategy.get_time_stamp()
