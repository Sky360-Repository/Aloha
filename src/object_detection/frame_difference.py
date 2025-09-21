# \copyright    Sky360.org
#
# \brief        Implementation of frame difference.
#
# ************************************************************************


import numpy as np
import warnings

class FrameDifference:
    def __init__(self, img_size, nbr_history_frame: np.uint8 = 2):
        img_height, img_width, img_channel = img_size
        self.img_height, self.img_width, self.img_channel = img_height, img_width, img_channel
        self.nbr_history_frame = nbr_history_frame
        if nbr_history_frame < 1:
            warnings.warn('Minimum number of history frames is 1.')
            self.nbr_history_frame = 1

        # Initialize parameters: shape = (img_height, img_width, img_channel, nbr_history_frame)
        self.frame_history = np.zeros((img_height, img_width, img_channel, self.nbr_history_frame), dtype=np.float32)


    def process_frame(self, image: np.ndarray, frame_dif_th: float = 0.1) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] != self.img_channel:
            raise ValueError(f"Expected {self.img_channel} channels, got {image.shape[2]}")

        image = (image.astype(np.float32) / 255.0) - 0.5
        image_exp = image[:, :, :, None]  # shape: (H, W, C, 1)

        # Distance to both history frames
        dist = (image_exp - self.frame_history)
        score = np.sqrt(np.sum(dist ** 2, axis=2))  # shape: (H, W, N)
        match_mask = np.any(score < frame_dif_th, axis=-1)  # shape: (H, W)
        fg_mask = ~match_mask

        # Update history
        for i in reversed(range(1, self.nbr_history_frame)):
            self.frame_history[:, :, :, i] = self.frame_history[:, :, :, i - 1]
        self.frame_history[:, :, :, 0] = image

        return fg_mask.astype(np.uint8) * 255
