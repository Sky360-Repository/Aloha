# \copyright    Sky360.org
#
# \brief        JPEG compression functions.
#
# ************************************************************************

import cv2
import numpy as np


def convert_image_to_byte_array(image, quality=90):
    # Ensure the image is in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Encode the image with quality control
    result, encoded_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Convert the encoded image to a byte array
    byte_array = bytes(encoded_img)

    return byte_array


def convert_byte_array_to_image(byte_array):
    # Convert byte array to numpy array
    np_array = np.frombuffer(byte_array, dtype=np.uint8)

    # Decode the numpy array as an image
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB.
    # When displaying an image loaded with OpenCV in matplotlib the channels' order will be incorrect.
    # The easiest way of fixing this is to use OpenCV to explicitly convert it back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
