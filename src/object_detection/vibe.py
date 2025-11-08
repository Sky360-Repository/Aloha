# \copyright    Sky360.org
#
# \brief        Implementation of Gaussian Mixture of models.
#
# ************************************************************************

import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
import math


class ViBe:
    def __init__(self, img_size, nbr_backgrounds: np.uint8 = 20, min_matches: np.uint8 = 2):
        img_height, img_width, img_channel = img_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel

        self.nbr_backgrounds = nbr_backgrounds
        self.min_matches = min_matches
        self.eps = 1e-6

        # Initialize background buffer: shape = (H, W, C, N)
        self.bg_buffer = np.full((img_height, img_width, img_channel, self.nbr_backgrounds), 2.0, dtype=np.float32)

        # Thread pool for asynchronous execution
        self.executor = ThreadPoolExecutor(max_workers=1)

    def get_difference_mask(self, image: np.ndarray, frame_dif_th: float = 0.2) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] != self.img_channel:
            raise ValueError(f"Expected {self.img_channel} channels, got {image.shape[2]}")

        # Normalize to [-1, 1]
        image = (image.astype(np.float32) / 255.0) - 0.5
        image_exp = image[:, :, :, None]  # shape: (H, W, C, 1)

        # Euclidean distance to each sample in the buffer
        dist = image_exp - self.bg_buffer  # shape: (H, W, C, N)
        score = np.sqrt(np.sum(dist ** 2, axis=2))  # shape: (H, W, N)

        # Count how many samples are within threshold
        match_count = np.sum(score < frame_dif_th, axis=-1)  # shape: (H, W)

        # Foreground if not enough matches
        fg_mask = match_count < self.min_matches

        return fg_mask.astype(np.uint8) * 255
    
    def update(self, image: np.ndarray, background_mask: np.ndarray = None, frame_dif_th: float = 0.1) -> Future:
        # Initialize future if first time
        if not hasattr(self, 'previous_future') or self.previous_future.done():
            # Submit async task only if the previous one is done (or doesn't exist)
            self.previous_future = self.executor.submit(
                self._threaded_update,
                image.copy(),
                background_mask.copy() if background_mask is not None else None,
                frame_dif_th
            )
        else:
            # Skip submission if previous task is still running
            pass  # Or log it / count skipped frames

        return self.previous_future

    def _threaded_update(self, image, background_mask, frame_dif_th):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] != self.img_channel:
            raise ValueError(f"Expected {self.img_channel} channels, got {image.shape[2]}")

        # Normalize to [-1, 1]
        image = (image.astype(np.float32) / 255.0) - 0.5

        # Compute match mask (optional, if you want to refine background_mask)
        image_exp = image[:, :, :, None]  # shape: (H, W, C, 1)
        dist = image_exp - self.bg_buffer  # shape: (H, W, C, N)
        score = np.sqrt(np.sum(dist ** 2, axis=2))  # shape: (H, W, N)
        match_count = np.sum(score < frame_dif_th, axis=-1)  # shape: (H, W)
        match_mask = match_count < self.min_matches

        # Combine with external background mask
        if background_mask is not None:
            background_mask = background_mask.astype(bool)
            update_mask = match_mask & background_mask
        else:
            update_mask = match_mask

        # Get indices of pixels to update
        y_idx, x_idx = np.where(update_mask)
        num_updates = len(y_idx)

        if num_updates == 0:
            return  # nothing to update

        # Random slots to update in buffer
        rand_slots = np.random.randint(0, self.nbr_backgrounds, size=num_updates)

        # Broadcast pixel values to buffer slots
        for c in range(self.img_channel):
            self.bg_buffer[y_idx, x_idx, c, rand_slots] = image[y_idx, x_idx, c]

            # Optional neighbor update
            if np.random.rand() < 0.5:
                dy = np.clip(y_idx + np.random.randint(-1, 2, size=num_updates), 0, self.img_height - 1)
                dx = np.clip(x_idx + np.random.randint(-1, 2, size=num_updates), 0, self.img_width - 1)
                rand_slots_n = np.random.randint(0, self.nbr_backgrounds, size=num_updates)
                self.bg_buffer[dy, dx, c, rand_slots_n] = image[y_idx, x_idx, c]