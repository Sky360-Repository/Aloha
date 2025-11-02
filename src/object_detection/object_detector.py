import cv2
import numpy as np
import argparse
import time

from skimage.feature import blob_dog
from skimage import morphology, measure, color, filters
from skimage.feature import shape_index
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from frame_difference import FrameDifference
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from gaussian_mix_models import GaussianMixModels
from vibe import ViBe

class ObjectDetector:
    @dataclass
    class Config:
        nbr_history_frame: int = 3
        min_obj_size: int = 30
        min_hole_area: int = 20
        max_coverage: float = 0.5
        mag_frame_dif_th: float = 0.1
        ang_frame_dif_th: float = 0.2
        rgb_frame_dif_th: float = 0.1
        nbr_gaussians: np.uint8 = 7
        learning_factor: np.float32 = 0.5
        nbr_backgrounds: np.uint8 = 20
        min_matches: np.uint8 = 2

    def __init__(self, rgb_img_size, config: Config = None):
        if config is None:
            config = ObjectDetector.Config()
        self.config = config

        gray_depth = 1
        img_height, img_width, img_depth = rgb_img_size
        self.fd_mag = FrameDifference((img_height, img_width, gray_depth), config.nbr_history_frame)
        self.fd_ang = FrameDifference((img_height, img_width, gray_depth), config.nbr_history_frame)
        self.fd_rgb = FrameDifference((img_height, img_width, img_depth), config.nbr_history_frame)
        self.config = config

        self.gmm_bg = GaussianMixModels((img_height, img_width, img_depth), self.config.nbr_gaussians, self.config.learning_factor)
        self.vibe_bg = ViBe((img_height, img_width, img_depth), self.config.nbr_backgrounds, self.config.min_matches)

    # TODO: Option to load parameters from json
    """
    @staticmethod
    def load_config_from_json(path: str) -> 'ObjectDetector.Config':
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return ObjectDetector.Config(**data)
    """
    def process_frame(self, rgb_image: np.ndarray):

        # Gaussian Blur - also for demo so that is faster to learn the background
        blured_img = cv2.GaussianBlur(rgb_image, (7, 7), 0)
        gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Features from extract_gradients
        grad_magnitude, grad_angle = self.extract_gradients(gray_img)

        with ThreadPoolExecutor() as executor:
            futures = {
                'mag': executor.submit(self.fd_mag.process_frame, grad_magnitude, self.config.mag_frame_dif_th),
                'ang': executor.submit(self.fd_ang.process_frame, grad_angle, self.config.ang_frame_dif_th),
                'rgb': executor.submit(self.fd_rgb.process_frame, rgb_image, self.config.rgb_frame_dif_th),
                'dog': executor.submit(self.detect_dots_dog, gray_img),
                'gmm': executor.submit(self.gmm_bg.get_difference_mask, blured_img),
                'vibe': executor.submit(self.vibe_bg.get_difference_mask, blured_img)
            }

        # Gradient based detector
        mag_mask = futures['mag'].result()
        mag_mask = self.validate_mask(mag_mask, self.config.max_coverage)

        ang_mask = futures['ang'].result()
        ang_mask = self.clean(ang_mask, self.config.min_obj_size, self.config.min_hole_area)

        # RGB frame difference
        rgb_diff_mask = futures['rgb'].result()

        # DoG detecting areas of interest
        blobs_dog_mask = futures['dog'].result()
        blobs_dog_mask = self.clean(blobs_dog_mask, self.config.min_obj_size)

        gmm_mask = futures['gmm'].result()
        gmm_mask = self.validate_mask(gmm_mask, self.config.max_coverage)
        vibe_mask = futures['vibe'].result()

        # TODO: add GMM and ViBe
        # Fusion logic (TBD):
        #   moving = fg_diff & (fg_vibe | fg_gmm)
        #   static = (fg_vibe & fg_gmm) & ~fg_diff
        #   ghosts = (fg_vibe ^ fg_gmm) & ~fg_diff
        #   background = ~(fg_vibe | fg_gmm | fg_diff)

        # foreground_obj = GMM_mask & Vibe_mask
        # self.moving_obj = mag_mask | ang_mask | rgb_diff_mask
        # TODO: add GMM and ViBe for static objects
        # static_obj = foreground_obj & ~moving_obj
        # TODO: add GMM and ViBe for foreground_obj
        # self.interest_obj = blobs_dog_mask # & foreground_obj
        self.mag_mask = mag_mask
        self.ang_mask = ang_mask
        self.rgb_diff_mask = rgb_diff_mask
        self.blobs_dog_mask = blobs_dog_mask
        self.gmm_mask = gmm_mask
        self.vibe_mask = vibe_mask

        # Update GMM and ViBe
        self.gmm_bg.update(blured_img)
        self.vibe_bg.update(blured_img)

    def validate_mask(self, mask: np.ndarray, max_coverage: float = 0.5) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D binary image.")

        total_pixels = mask.size
        active_pixels = np.count_nonzero(mask)
        coverage_ratio = active_pixels / total_pixels

        if coverage_ratio > max_coverage:
            return np.zeros_like(mask, dtype=np.uint8)
        else:
            return mask

    def extract_gradients(self, gray_img: np.ndarray, ksize: np.uint8 = 3) -> np.ndarray:
        # Gaussian Filter
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

        # Gradients
        grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize)
        grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize)

        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=False)
        angle = np.mod(angle, np.pi) # edges are symmetric

        magnitude = 255.0 * cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        angle = 255.0 * angle / np.pi  # Normalize angle to [0, 255]

        # Remove noise that span 80 pixels:
        # sigmaX ≈ 80 / 2.8 ≈ 29
        angle = cv2.GaussianBlur(angle, (0, 0), sigmaX = 29)

        return magnitude, angle

    def detect_dots_dog(self, gray_image, sigma1: float = 2.0, sigma2: float = 4.0):
        # For a Gaussian blob, the effective diameter D
        # D ≈ 2 * sqrt(2) * sigma ≈ 2.8 * sigma
        # To detect stars that span 5 pixels, you want:
        # sigma1 ≈ 5 / 2.8 ≈ 2 and sigma2 ≈ 2 * sigma1 ≈ 4

        # Ensure float32 for precision
        gray = gray_image.astype(np.float32) / 255.0

        # Apply fast Gaussian blurs
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)

        # Use contrast of blurred images to decide polarity
        is_high_contrast = (blur1.max() - blur1.min()) > (blur2.max() - blur2.min())

        # Difference of Gaussians
        dog = blur2 - blur1 if is_high_contrast else blur1 - blur2

        # Threshold using Otsu
        thresh = filters.threshold_otsu(dog)
        bw_img = (dog > thresh).astype(np.uint8) * 255

        return bw_img


    def clean(self, blob_bin_img, min_obj_size: np.uint8 = 30, min_hole_area: np.uint8 = 20):
        # We want to keep blobs are 3–6 pixels in diameter
        # Recommended: min_obj_size = 10–30
        # Use 10 if you want to keep faint stars.
        # Use 30+ if you want to suppress noise and isolate larger blobs.
        # Recommended: min_hole_area = 5–20
        # Use 5 to fill small gaps.
        # Use 20+ only if blobs are large and ring-like.

        # binary mask
        masks_bin = blob_bin_img.astype(bool)

        # Morphological cleaning (remove small noise and fill holes)
        cleaned = morphology.remove_small_objects(masks_bin, min_size=min_obj_size)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_hole_area)
        cleaned = morphology.binary_closing(cleaned, morphology.disk(3))
        cleaned = morphology.binary_opening(cleaned, morphology.disk(2))
        cleaned = cleaned.astype(np.uint8) * 255

        return cleaned

    def get_objerts_list(self, blob_bin_img):
        # Region properties using skimage
        labels = measure.label(blob_bin_img)
        regions = measure.regionprops(labels)

        object_dict = {}
        for i, region in enumerate(regions):
            if region.area < 200:
                continue

            y, x = map(int, region.centroid)
            object_dict[i] = {
                "centroid": (x, y),
                "area": region.area,
                "eccentricity": region.eccentricity,
                "bbox": region.bbox,
            }
        return object_dict