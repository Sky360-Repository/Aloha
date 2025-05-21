# \copyright    Sky360.org
#
# \brief        Strategy design to handle Image massages.
#
# ************************************************************************

import numpy as np
from abc import ABC, abstractmethod

from ecal_util.jpeg_compression import convert_byte_array_to_image


# Define the interface for strategies
class ImageStrategy(ABC):
    @abstractmethod
    def process_message(self, message):
        pass


# Concrete strategy for web_camera_image
class WebCameraImageStrategy(ImageStrategy):
    def __init__(self):
        self.time_stamp = 0
        self.rgb = None

    def process_message(self, message):
        if message.jpeg_size == 0:
            # Convert the raw image data back into an image
            frame = np.frombuffer(message.raw_image, dtype=np.uint8)
            self.rgb = frame.reshape((message.height, message.width, 3))
        else:
            # jpeg decoder
            self.rgb = convert_byte_array_to_image(message.jpeg_data)

        self.time_stamp = message.time_stamp

    def get_rgb_image(self):
        return self.rgb

    def get_time_stamp(self):
        return self.time_stamp


# Concrete strategy for CameraImage
class CameraImageStrategy(ImageStrategy):
    def __init__(self):
        self.time_stamp = 0
        self.rgb = None

    def process_message(self, message):
        if message.jpeg_size == 0:
            # Convert the raw image data back into an image
            frame = np.frombuffer(message.raw_image, dtype=np.uint8)
            self.rgb = frame.reshape((message.height, message.width, 3))
        else:
            # jpeg decoder
            self.rgb = convert_byte_array_to_image(message.jpeg_data)

        # Convert both to uint64
        u_sec = np.uint64(message.t_Stamp.u_sec)
        u_nano_sec = np.uint64(message.t_Stamp.u_nano_sec)

        # Convert seconds to nanoseconds and add the nanoseconds part
        self.time_stamp = np.uint64((u_sec * 1e9) + u_nano_sec)

    def get_rgb_image(self):
        return self.rgb

    def get_time_stamp(self):
        return self.time_stamp


# Context class
class ImageMessage:
    def __init__(self, message: str):
        if message == "web_camera_image":
            self._strategy = WebCameraImageStrategy()

        elif message == "CameraImage":
            self._strategy = CameraImageStrategy()

        else:
            Exception("Invalid message type. Please check that correct proto_file is used.")

    def process_message(self, message):
        return self._strategy.process_message(message)

    def get_rgb_image(self):
        return self._strategy.get_rgb_image()

    def get_time_stamp(self):
        return self._strategy.get_time_stamp()
