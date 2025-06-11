# \copyright    Sky360.org
#
# \brief        Implementation of Gaussian Mixture of models.
#
# ************************************************************************


import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
import math

class GaussianMixModels:
    def __init__(self, img_size, nbr_gaussians: np.uint8 = 7, learning_factor: np.float32 = 0.5):
        img_height, img_width, img_channel = img_size
        self.img_height, self.img_width, self.img_channel = img_height, img_width, img_channel
        self.nbr_gaussians = nbr_gaussians
        if nbr_gaussians < 3:
            warnings.warn('Minimum number of Gaussians is 3.')
            self.nbr_gaussians = 3
        self.alpha = np.clip(learning_factor, 1e-4, 1.0)
        self.eps = 1e-6
        self.initial_std = 0.01  # 2.5 / 255 based on noise and quantization
        self.min_std = 0.01  # 2.5 / 255 based on noise and quantization

        # Number of current possible backgrounds
        self.nbr_curr_backgrounds = np.minimum(3, np.floor(self.nbr_gaussians / 2))

        # Initialize GMM parameters: shape = (img_height, img_width, img_channel, nbr_gaussians)
        self.mean = np.zeros((img_height, img_width, img_channel, self.nbr_gaussians), dtype=np.float32)
        self.std = np.full((img_height, img_width, img_channel, self.nbr_gaussians), self.initial_std, dtype=np.float32)
        self.weight = np.full((img_height, img_width, self.nbr_gaussians), 1.0 / self.nbr_gaussians, dtype=np.float32)

        # Thread pool for asynchronous execution
        self.executor = ThreadPoolExecutor(max_workers=1)

        print(f"Estimated frames to integrate ghosts: {self.get_update_rate()} frames (target 70% weight)")


    def update(self, image: np.ndarray, background_mask: np.ndarray = None, std_factor: float = 2.0) -> Future:
        # Initialize future if first time
        if not hasattr(self, 'previous_future') or self.previous_future.done():
            # Submit async task only if the previous one is done (or doesn't exist)
            self.previous_future = self.executor.submit(
                self._threaded_update,
                image.copy(),
                background_mask.copy() if background_mask is not None else None,
                std_factor
            )
        else:
            # Skip submission if previous task is still running
            pass  # Or log it / count skipped frames

        return self.previous_future



    def _threaded_update(self, image, background_mask, std_factor):
        image = (image.astype(np.float32) / 255.0) - 0.5
        if self.img_channel == 1:
            image_exp = image[:, :, None, None]  # (img_height, img_width, img_channel, 1)
        else:
            image_exp = image[:, :, :, None]  # (img_height, img_width, img_channel, 1)

        dist = np.abs(image_exp - self.mean) / (self.std + self.eps)
        score = np.sqrt(np.sum(dist ** 2, axis=2))  # (img_height, img_width, nbr_gaussians)

        best_score = np.min(score, axis=-1)
        match_idx = np.argmin(score, axis=-1)
        match_mask = best_score < std_factor

        if background_mask is not None:
            background_mask = background_mask.astype(bool)
            match_mask &= background_mask
        else:
            background_mask = np.ones_like(match_mask, dtype=bool)

        match_idx[~match_mask] = -1
        valid_mask = match_idx >= 0

        one_hot = np.zeros((self.img_height, self.img_width, self.nbr_gaussians), dtype=np.float32)
        one_hot[valid_mask] = np.eye(self.nbr_gaussians)[match_idx[valid_mask]]
        one_hot_exp = one_hot[:, :, None, :]

        # Update matched Gaussians
        self.mean = (1 - self.alpha * one_hot_exp) * self.mean + (self.alpha * one_hot_exp) * image_exp
        var = self.std ** 2
        var = (1 - self.alpha * one_hot_exp) * var + (self.alpha * one_hot_exp) * (image_exp - self.mean) ** 2
        self.std = np.maximum(np.sqrt(var), self.min_std)
        self.weight = (1 - self.alpha) * self.weight + self.alpha * one_hot

        # Initialize unmatched background pixels
        unmatched_mask = (match_idx == -1) & background_mask
        if np.any(unmatched_mask):
            weakest_idx = np.argmin(self.weight, axis=-1)
            replace_mask = np.eye(self.nbr_gaussians)[weakest_idx] * unmatched_mask[..., None]
            replace_mask_exp = replace_mask[:, :, None, :]

            self.mean = np.where(replace_mask_exp, image_exp, self.mean)
            self.std = np.where(replace_mask_exp, self.initial_std, self.std)
            self.weight = np.where(replace_mask, 1e-3, self.weight)

        # Normalize weights
        weight_sum = np.sum(self.weight, axis=-1, keepdims=True) + self.eps
        self.weight /= weight_sum

        # Sort Gaussians by weight
        sorted_idx = np.argsort(-self.weight, axis=-1)
        idx_exp_c = sorted_idx[:, :, np.newaxis, :]
        self.mean = np.take_along_axis(self.mean, idx_exp_c, axis=-1)
        self.std = np.take_along_axis(self.std, idx_exp_c, axis=-1)
        self.weight = np.take_along_axis(self.weight, sorted_idx, axis=-1)

    def get_background(self):
        bg = np.clip((self.mean[:, :, :, 0] + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return bg

    def get_difference_mask(self, image: np.ndarray, std_factor: float = 2.5) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] != self.img_channel:
            raise ValueError(f"Expected {self.img_channel} channels, got {image.shape[2]}")

        image = (image.astype(np.float32) / 255.0) - 0.5
        image_exp = image[:, :, :, None]  # (img_height, img_width, img_channel, 1)

        # Compute Mahalanobis distance to the first N gaussians
        N = int(self.nbr_curr_backgrounds)
        mean = self.mean[:, :, :, :N]  # (img_height, img_width, img_channel, N)
        std = self.std[:, :, :, :N]    # (img_height, img_width, img_channel, N)

        dist = (image_exp - mean) / (std + self.eps)
        score = np.sqrt(np.sum(dist ** 2, axis=2))  # (img_height, img_width, N)

        # Check if any of the top N match
        match_mask = np.any(score < std_factor, axis=-1)  # (img_height, img_width)
        fg_mask = ~match_mask

        return (fg_mask.astype(np.uint8)) * 255

    def get_update_rate(self, target_weight: float = 0.7) -> int:
        if not (0 < target_weight < 1):
            raise ValueError("target_weight must be between 0 and 1.")

        # Avoid log(0) or division by 0
        if self.alpha >= 1.0:
            return 1

        try:
            n_frames = math.ceil(math.log(1 - target_weight) / math.log(1 - self.alpha))
            return n_frames
        except ZeroDivisionError:
            return float('inf')  # Unreachable if alpha is 0 (shouldn’t happen)

