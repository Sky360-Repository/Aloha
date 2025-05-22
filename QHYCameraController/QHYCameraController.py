#!/usr/bin/env python3
# coding: utf-8

import subprocess
import time
import numpy as np
import cv2
import ctypes
from ctypes import *
from multiprocessing import shared_memory

# Time formatting -------------------------------------------
def format_timestamp_utc(ts: float) -> str:
    #Format a UNIX timestamp to a UTC time string with milliseconds
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}"

class RingBuffer:
    def __init__(self, roi_width: int, roi_height: int, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.frame_bytes = roi_width * roi_height  # Number of pixels per frame
        self.frame_index = 0  # Start index for buffer cycling

        # Shared memory for image frames
        self.shm = shared_memory.SharedMemory(create=True, size=self.buffer_size * self.frame_bytes * 2, name="cam_ring_buffer")
        self.frame_buffer = np.ndarray((self.buffer_size, roi_height, roi_width), dtype=np.uint16, buffer=self.shm.buf)

        # Shared memory for timestamps
        timestamp_bytes = self.buffer_size * np.dtype(np.float64).itemsize
        self.shm2 = shared_memory.SharedMemory(create=True, size=timestamp_bytes, name="metadata_ring_buffer")
        self.timestamp_buffer = np.ndarray((self.buffer_size,), dtype=np.float64, buffer=self.shm2.buf)

    def store_frame(self, img: np.ndarray):
        # Stores a frame in the ring buffer and updates timestamp.
        self.frame_buffer[self.frame_index] = img
        self.timestamp_buffer[self.frame_index] = time.time()
        self.frame_index = (self.frame_index + 1) % self.buffer_size  # Cycle index

    def get_latest_frame(self):
        # Retrieves the most recent frame from the buffer.
        return self.frame_buffer[(self.frame_index - 1) % self.buffer_size]

    def cleanup(self):
        # Releases shared memory resources.
        self.shm.close()
        self.shm.unlink()
        self.shm2.close()
        self.shm2.unlink()

# AE Controller
class AEController:
    # Constants
    TARGET_GAIN = 22  # Initial gain
    TARGET_DARK = 0.05  # [0..1]
    TARGET_BRIGHT = 0.20  # [0..1]
    EXPOSURE_MIN = 0  # [µs]
    EXPOSURE_MAX = 140000  # [µs]
    MIN_GAIN = 1.0
    MAX_GAIN = 59.0
    GAIN_ADJUSTMENT_SPEED = 0.02  # How fast gain is adjusted
    SMOOTHED_ALPHA = 0.2  # [0..1]
    HYSTERESIS = 0.015  # [0..1]
    HYSTERESIS_BAND = 0.05  # Smooth scaling within ±5% error
    MAX_EXP_STEP = 5000  # [µs]
    MIN_EXP_STEP = 100  # [µs]
    MAX_GAIN_STEP = 0.5
    MIN_GAIN_STEP = 0.05

    def __init__(self):
        self.state = "adjust_exposure"
        self.smoothed_dark_ratio = None
        self.smoothed_bright_ratio = None
        self.error_ratio = 0
        self.brightness_error = 0.1

    def adaptive_step(self, error, max_step, min_step, scale_factor):
        scale = min(1.0, abs(error) / self.HYSTERESIS_BAND)
        return min(max_step, max(min_step, error * scale_factor * scale))

    def aggressive_adaptive_step(self, error, max_step, min_step, scale_factor):
        scale = min(1.0, abs(error) / self.HYSTERESIS_BAND)
        return min(max_step, max(min_step, error * scale_factor * (scale ** 1.5)))

    def ema_smoothing(self, frame_buffer, frame_index):
        # Computes the histogram, CDF, and smooths brightness values.
        gray_img = (frame_buffer[frame_index] / 65535.0).astype(np.float32)
        histogram = cv2.calcHist([gray_img], [0], None, [1024], [0.0, 1.0]).flatten()
        histogram /= histogram.sum()
        cdf = np.clip(np.cumsum(histogram), 0.0, 1.0)
        dark_ratio = cdf[10]
        bright_ratio = 1.0 - cdf[1014]

        if self.smoothed_dark_ratio is None:
            self.smoothed_dark_ratio = dark_ratio
            self.smoothed_bright_ratio = bright_ratio
        else:
            self.smoothed_dark_ratio = self.SMOOTHED_ALPHA * dark_ratio + (1 - self.SMOOTHED_ALPHA) * self.smoothed_dark_ratio
            self.smoothed_bright_ratio = self.SMOOTHED_ALPHA * bright_ratio + (1 - self.SMOOTHED_ALPHA) * self.smoothed_bright_ratio

    def predictive_jump_estimate(self, exposure_us, error_magnitude):
        # Computes predictive exposure adjustment based on error magnitude.
        if error_magnitude > 0.2:
            target_luminance = (self.TARGET_DARK + self.TARGET_BRIGHT) / 2.0
            self.error_ratio = self.smoothed_dark_ratio - target_luminance
            predicted_exposure = exposure_us * (1 + 2.5 * self.error_ratio)
            return np.clip(predicted_exposure, self.EXPOSURE_MIN, self.EXPOSURE_MAX)
        return exposure_us

    def adjust_exposure(self, exposure_us, dark_error, bright_error):
        # Adjusts exposure dynamically.
        if dark_error > self.HYSTERESIS:
            desired_exposure = exposure_us + self.aggressive_adaptive_step(dark_error, self.MAX_EXP_STEP, self.MIN_EXP_STEP, 80000)
        elif bright_error > self.HYSTERESIS:
            desired_exposure = exposure_us - self.aggressive_adaptive_step(bright_error, self.MAX_EXP_STEP, self.MIN_EXP_STEP, 80000)
        else:
            desired_exposure = exposure_us

        new_exposure = max(self.EXPOSURE_MIN, min(desired_exposure, self.EXPOSURE_MAX))
        return new_exposure, abs(new_exposure - exposure_us) > 1  # Return adjusted exposure and flag

    def adjust_gain(self, gain, dark_error, bright_error):
        # Adjusts gain dynamically.
        if dark_error > self.HYSTERESIS and gain < self.MAX_GAIN:
            gain += self.aggressive_adaptive_step(dark_error, self.MAX_GAIN_STEP, self.MIN_GAIN_STEP, 200)
        elif bright_error > self.HYSTERESIS and gain > self.MIN_GAIN:
            gain -= self.aggressive_adaptive_step(bright_error, self.MAX_GAIN_STEP, self.MIN_GAIN_STEP, 80)
        elif abs(self.brightness_error) < 0.015 and abs(self.smoothed_bright_ratio - self.TARGET_BRIGHT) < 0.015:
            gain += (self.TARGET_GAIN - gain) * self.GAIN_ADJUSTMENT_SPEED

        return min(max(gain, self.MIN_GAIN), self.MAX_GAIN)

    def update(self, exposure_us, gain, frame_buffer, frame_index):
        # Main update loop handling exposure and gain adaptation.
        self.ema_smoothing(frame_buffer, frame_index)

        dark_error = self.smoothed_dark_ratio - self.TARGET_DARK
        bright_error = self.smoothed_bright_ratio - self.TARGET_BRIGHT
        self.brightness_error = 0.5 * dark_error + 0.5 * (-bright_error)
        error_magnitude = abs(self.brightness_error)

        # Early exit if within dead zone
        if abs(dark_error) < 0.005 and abs(bright_error) < 0.005:
            return exposure_us, gain

        # Predictive adjustment for large errors
        exposure_us = self.predictive_jump_estimate(exposure_us, error_magnitude)

        # Exposure adjustment
        if self.state == "adjust_exposure":
            exposure_us, exposure_adjusted = self.adjust_exposure(exposure_us, dark_error, bright_error)
            if exposure_adjusted:
                return exposure_us, gain
            else:
                self.state = "adjust_gain"

        # Gain adjustment
        if self.state == "adjust_gain":
            gain = self.adjust_gain(gain, dark_error, bright_error)
            if self.EXPOSURE_MIN < exposure_us < self.EXPOSURE_MAX:
                self.state = "adjust_exposure"

        return exposure_us, gain

class QHYCameraController:
    # Constants
    TARGET_GAIN = 22 # initial gain
    TARGET_EXPOSURE = 30000 # [µs] initial exposure
    CONTROL_EXPOSURE = 8
    CONTROL_GAIN = 6
    CONTROL_TRANSFERBIT = 10
    CONTROL_USBTRAFFIC = 12
    CONTROL_CURTEMP = 14
    CONTROL_CURPWM = 15
    CONTROL_MANULPWM = 16
    CONTROL_COOLER = 18
    TARGET_TEMP = -20.0 # [°C]
    TEMPERATURE_TOLERANCE = 1.0 # [°C]

    ROI_WIDTH = ROI_HEIGHT = 3200 # squared ROI, centered to zenith
    FPS_TARGET = 7 # target FPS
    BITS_PER_PIXEL = 16  # Image depth (bit-depth)
    COLOR_CHANNELS = 1  # Number of image color channels (monochrome)

    MIN_DELAY_SEC = 0.03 # [s] floor for delay estimation
    MIN_GRAB_TIME_SEC = 0.102 # [s] minimum grabbing estimate for first delay
    RING_BUFFER_FRAMES = 200  # number of frames in ring buffer
    MAX_FRAME_GRABS = 10 # max number of successful frame grabs
    EXPOSURE_ADJUST_INTERVAL = 5 # Frequency of exposure/gain adjustments

    # Buffer & Processing Parameters
    FRAME_SYNC_DELAY_STEP = -0.001  # Decreasing delay step with new frame grabs [seconds]
    FRAME_GRAB_PENALTY_SEC = 0.31  # Forced sleep in case of failed frame grab [seconds]

    def __init__(self, sdk_path='/usr/local/lib/libqhyccd.so'):
        self.bpp = c_uint32(self.BITS_PER_PIXEL)
        self.channels = c_uint32(self.COLOR_CHANNELS)
        self.ring_buffer = RingBuffer(self.ROI_WIDTH, self.ROI_HEIGHT)  # 🚀 Initialize RingBuffer
        self.x_offset = int((5544 - self.ROI_WIDTH) / 2) # [px]
        self.y_offset = int((3684 - self.ROI_HEIGHT) / 2) # [px]
        self.cycle = 1 / self.FPS_TARGET # [s]
        self.delay = max(self.MIN_DELAY_SEC, self.cycle - self.MIN_GRAB_TIME_SEC) # [s]
        self.frame_index = 0
        self.w = c_uint32(self.ROI_WIDTH)
        self.h = c_uint32(self.ROI_HEIGHT)
        self.successes = 0
        self.consecutive_failures = 0

        # Auto-exposure
        self.ae_controller = AEController()

        # Loading SDK
        self.sdk = CDLL(sdk_path)
        self._set_function_signatures()
        self.cam = None
        self.initialize_camera()
        self.set_exposure_and_gain()

        # Create the ring buffer
        self.create_ring_buffer()

        # Calibration
        img = self.get_single_frame()
        self.set_live_mode()
        self.calibrate_camera()

    def _set_function_signatures(self):
        # type_char_array_32 = c_char*32
        self.sdk.InitQHYCCDResource.restype = c_uint32
        self.sdk.GetQHYCCDId.argtypes = [c_uint32, c_char_p]
        self.sdk.GetQHYCCDId.restype = c_uint32
        self.sdk.OpenQHYCCD.argtypes = [c_char_p]
        self.sdk.OpenQHYCCD.restype = ctypes.POINTER(c_uint32)
        self.sdk.InitQHYCCD.argtypes = [c_void_p]
        self.sdk.InitQHYCCD.restype = c_uint32
        self.sdk.SetQHYCCDStreamMode.argtypes = [c_void_p, c_uint8]
        self.sdk.SetQHYCCDStreamMode.restype = c_uint32
        self.sdk.SetQHYCCDParam.argtypes = [c_void_p, c_uint32, c_double]
        self.sdk.SetQHYCCDParam.restype = c_uint32
        self.sdk.GetQHYCCDParam.argtypes = [c_void_p, c_uint32]
        self.sdk.GetQHYCCDParam.restype = c_double
        self.sdk.GetQHYCCDMemLength.argtypes = [c_void_p]
        self.sdk.GetQHYCCDMemLength.restype = c_uint32
        self.sdk.GetQHYCCDLiveFrame.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(ctypes.c_uint16)]
        self.sdk.GetQHYCCDLiveFrame.restype = c_uint32
        self.sdk.BeginQHYCCDLive.argtypes = [c_void_p]
        self.sdk.BeginQHYCCDLive.restype = c_uint32
        self.sdk.ExpQHYCCDSingleFrame.argtypes = [c_void_p]
        self.sdk.ExpQHYCCDSingleFrame.restype = c_uint32
        self.sdk.GetQHYCCDSingleFrame.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(ctypes.c_uint16)]
        self.sdk.GetQHYCCDSingleFrame.restype = c_uint32
        self.sdk.CloseQHYCCD.argtypes = [c_void_p]
        self.sdk.CloseQHYCCD.restype = c_uint32
        self.sdk.ReleaseQHYCCDResource.restype = c_uint32
        self.sdk.SetQHYCCDResolution.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_uint32]
        self.sdk.SetQHYCCDResolution.restype = c_uint32
        self.sdk.SetQHYCCDBitsMode.argtypes = [c_void_p, c_uint32]
        self.sdk.SetQHYCCDBitsMode.restype = c_uint32
        self.sdk.StopQHYCCDLive.argtypes = [c_void_p]
        self.sdk.StopQHYCCDLive.restype = c_uint32

    def initialize_camera(self):
        # Resetting USB
        try:
            subprocess.check_call(["usbreset", "Q183-Cool"])
        except subprocess.CalledProcessError as e:
            exit(0)
        time.sleep(2.0)

        # Initializing SDK
        self.sdk.InitQHYCCDResource()
        time.sleep(2)

        # Scanning for cameras
        self.sdk.ScanQHYCCD()
        time.sleep(0.2)

        # Getting first camera ID
        id_buffer = create_string_buffer(32)
        self.sdk.GetQHYCCDId(0, id_buffer)
        cam_id = id_buffer.value.decode().strip('\x00')

        # Opening camera
        self.cam = self.sdk.OpenQHYCCD(id_buffer)

        # Setting 16-bit mode
        self.sdk.SetQHYCCDBitsMode(self.cam, 16)
        time.sleep(0.2)

        # Setting transfer bit
        self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT, 16)
        tbit = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT)
        time.sleep(0.2)

        # Setting resolution
        self.sdk.SetQHYCCDResolution(
            self.cam,
            c_uint(self.x_offset),
            c_uint(self.y_offset),
            c_uint(self.ROI_WIDTH),
            c_uint(self.ROI_HEIGHT)
        )
        time.sleep(0.2)

        # Setting USB traffic to 0
        self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_USBTRAFFIC, 0)
        time.sleep(0.2)

        # Initializing camera
        self.sdk.InitQHYCCD(self.cam)
        time.sleep(2)

    def set_exposure_and_gain(self):
        # Setting exposure and gain
        self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE, self.TARGET_EXPOSURE)
        self.exposure = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE)

        self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.TARGET_GAIN)
        self.gain = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_GAIN)
        time.sleep(0.2)

    def set_live_mode(self):
        # Setting mode for live frames
        ret = self.sdk.SetQHYCCDStreamMode(self.cam, 1)
        time.sleep(1.0)

        # Setting resolution
        ret = self.sdk.SetQHYCCDResolution(self.cam, c_uint(self.x_offset), c_uint(self.y_offset), c_uint(self.ROI_WIDTH), c_uint(self.ROI_HEIGHT))
        time.sleep(0.2)

        # Begin live
        ret = self.sdk.BeginQHYCCDLive(self.cam)
        time.sleep(1.0)

    def create_ring_buffer(self):
        # Get allocated memory size
        ddr_bytes = self.sdk.GetQHYCCDMemLength(self.cam)
        frame_bytes = self.ROI_WIDTH * self.ROI_HEIGHT  # Number of pixels per frame

        # Create shared memory block for image data
        shm = shared_memory.SharedMemory(create=True, size=self.RING_BUFFER_FRAMES * frame_bytes * 2, name="cam_ring_buffer")
        frame_buffer = np.ndarray((self.RING_BUFFER_FRAMES, self.ROI_HEIGHT, self.ROI_WIDTH), dtype=np.uint16, buffer=shm.buf)

        # Create shared memory block for metadata
        timestamp_bytes = self.RING_BUFFER_FRAMES * np.dtype(np.float64).itemsize
        shm2 = shared_memory.SharedMemory(create=True, size=timestamp_bytes, name="metadata_ring_buffer")
        timestamp_buffer = np.ndarray((self.RING_BUFFER_FRAMES,), dtype=np.float64, buffer=shm2.buf)

        # Temporary buffer for single frame retrieval
        temp_buffer = (ctypes.c_uint16 * frame_bytes)()

        time.sleep(2.0)  # Allow initialization delay

        return frame_buffer, timestamp_buffer, temp_buffer, shm, shm2

    def get_single_frame(self):
        img = None  # Default to None
        for i in range(3):
            ret = self.sdk.ExpQHYCCDSingleFrame(self.cam)
            time.sleep(0.5)
            ret = self.sdk.GetQHYCCDSingleFrame(self.cam, byref(self.w), byref(self.h),
                                                byref(self.bpp), byref(self.channels), self.buffer)
            if ret == 0:
                img = np.frombuffer(self.buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
            time.sleep(1.5)
        return img

    def calibrate_camera(self):
        delay = self.MIN_DELAY_SEC
        successes = 0
        consecutive_failures = 0

        while successes < self.MAX_FRAME_GRABS:
            loop_start = time.perf_counter()

            # Try fetching a frame
            ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.buffer)
            grab_time = time.perf_counter() - loop_start

            if ret == 0:
                consecutive_failures = 0
                img_start = time.perf_counter()
                img = np.frombuffer(self.buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
                processing_time = time.perf_counter() - img_start
                duration = grab_time + processing_time
                self.speed = self.frame_bytes / duration
                self.successes += 1
                self.estimated_fps = 1.0 / duration

                # Update delay dynamically
                grab_and_process_time = time.perf_counter() - loop_start
                delay = max(self.MIN_DELAY_SEC, self.cycle - grab_and_process_time)
            else:
                consecutive_failures += 1
                time.sleep(self.FRAME_GRAB_PENALTY_SEC)

                if consecutive_failures >= 5:
                    time.sleep(self.FRAME_GRAB_PENALTY_SEC * 2)

                    # Stop & restart live camera to resync
                    self.sdk.StopQHYCCDLive(self.cam)
                    time.sleep(1.0)
                    self.sdk.BeginQHYCCDLive(self.cam)
                    time.sleep(1.0)

                    consecutive_failures = 0

                delay += self.FRAME_SYNC_DELAY_STEP
                delay += self.FRAME_SYNC_DELAY_STEP
                delay_delta = self.FRAME_SYNC_DELAY_STEP if delay not in (self.MIN_DELAY_SEC, self.cycle) else -self.FRAME_SYNC_DELAY_STEP
                successes -= 1
            time.sleep(delay)
        return delay

    # Temperature control
    def control_temperature_pwm(self, frame_index: int):
        if frame_index in {7, 107}:
            current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
            current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)

            if current_temp < -100:
                print("Warning: Invalid temperature reading, retrying...")
                time.sleep(2.0)  # Blocking, consider logging instead
                current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
                current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)

            if abs(current_temp - self.TARGET_TEMP) > self.TEMPERATURE_TOLERANCE:
                self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_COOLER, self.TARGET_TEMP)

    def get_live_frame(self):
        ret = self.sdk.GetQHYCCDLiveFrame(self.cam,
                                          ctypes.byref(self.w),
                                          ctypes.byref(self.h),
                                          ctypes.byref(self.bpp),
                                          ctypes.byref(self.channels),
                                          self.buffer)

        if ret == 0:
            # Convert buffer to image
            img = np.frombuffer(self.buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
            self.ring_buffer.store_frame(img)  # 🚀 Store frame in ring buffer

            # Auto-exposure correction at intervals
            if self.ring_buffer.frame_index % self.EXPOSURE_ADJUST_INTERVAL == 0:
                prev_exposure, prev_gain = self.exposure, self.gain

                # Compute new exposure and gain using the ring buffer
                self.exposure, self.gain = self.ae_controller.update(self.exposure,
                                                                     self.gain,
                                                                     self.ring_buffer.frame_buffer,
                                                                     self.ring_buffer.frame_index)

                # Apply changes if significant
                if abs(self.exposure - prev_exposure) > 1:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE, self.exposure)
                    self.exposure = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE)

                if abs(self.gain - prev_gain) > 0.05:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.gain)
                    self.gain = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_GAIN)

            # Perform temperature control
            self.control_temperature_pwm(self.ring_buffer.frame_index)

            return img  # Return the processed frame
        return None

    def close(self):
        self.ring_buffer.cleanup()

        # Stop live camera
        try:
            self.sdk.StopQHYCCDLive(self.cam)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error stopping live camera: {e}")

        # Stop cooler
        try:
            self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_MANULPWM, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error stopping cooler: {e}")

        # Release SDK resources
        self.sdk.ReleaseQHYCCDResource()
        time.sleep(0.2)

        # Close camera connection
        self.sdk.CloseQHYCCD(self.cam)
        time.sleep(2.0)

        # Reset USB
        try:
            subprocess.check_call(["usbreset", "Q183-Cool"])
            print("Resetting USB: ✅")
        except subprocess.CalledProcessError as e:
            print("Resetting USB: ❌", e)
        time.sleep(1.0)


if __name__ == "__main__":

    qhy_camera = QHYCameraController()

    # Live loop
    while True:
        img = qhy_camera.get_live_frame()

        if img is not None:
            # optional: for viewing (downsampled)
            debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
            scaled_img = debayered_img[::4, ::4]
            cv2.imshow("Image", scaled_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close
    qhy_camera.close()

    exit(0)
