#!/usr/bin/env python3
# coding: utf-8

import os
import subprocess
import time
import numpy as np
import cv2
import ctypes
import keyboard
from ctypes import *
from multiprocessing import shared_memory
import h5py
import threading
from numba import njit

debug = True
data_dump = False
quit_requested = False

# Catching the ENTER key to exit
def listen_for_key():
    global quit_requested
    input()
    quit_requested = True

threading.Thread(target=listen_for_key, daemon=True).start()
if debug: print(f"⏹️  Press ENTER to quit. ⏹️")

# Time formatting -------------------------------------------
def format_timestamp_utc(ts: float) -> str:
    #Format a UNIX timestamp to a UTC time string with milliseconds
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}"

class RingBuffer:
    def __init__(self, roi_width: int, roi_height: int, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.frame_bytes = roi_width * roi_height  # Number of pixels per frame
        self.frame_index = 0  # Start index for buffer cycling

        # Create shared memory block for image data
        self.shm = shared_memory.SharedMemory(create=True, size=self.buffer_size * self.frame_bytes * 2, name="cam_ring_buffer")
        self.frame_buffer = np.ndarray((self.buffer_size, roi_height, roi_width), dtype=np.uint16, buffer=self.shm.buf)

        # Create shared memory block for metadata
        metadata_bytes = np.prod((self.buffer_size, 3)) * np.dtype(np.float64).itemsize
        self.shm2 = shared_memory.SharedMemory(create=True, size=metadata_bytes, name="metadata_ring_buffer")
        self.metadata_buffer = np.ndarray((self.buffer_size, 3), dtype=np.float64, buffer=self.shm2.buf)

        if debug: print(f"RingBuffer: successfully created ring buffers {self.frame_buffer.shape} and {self.metadata_buffer.shape} ✅")

    def export_to_hdf5(self, filepath: str):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("frames", data=self.frame_buffer, compression="gzip")
            f.create_dataset("metadata", data=self.metadata_buffer, compression="gzip")
            f.flush()
            #os.fsync(f.fid.get_vfd_handle())
        if debug: print(f"RingBuffer: exported and flushed to {filepath} ✅")

    def store_frame(self, img: np.ndarray, exposure: float, gain: float):
        self.frame_buffer[self.frame_index] = img
        self.metadata_buffer[self.frame_index] = [time.time(), exposure, gain]

        if data_dump:
            # Writing the buffers into an HDF5 file for testing
            if self.frame_index == self.buffer_size - 1:
                timestamp = format_timestamp_utc(time.time()).replace(':', '-').replace(' ', '_')
                filename = f"/tmp/qhy_capture_{timestamp}.hdf5"
                self.export_to_hdf5(filename)

    def cleanup(self):
        # Releases shared memory resources.
        self.shm.close()
        self.shm.unlink()
        self.shm2.close()
        self.shm2.unlink()

# AE Controller
class AEController:
    SMOOTHING_ALPHA = 0.3

    def __init__(self, mask=None):
        self.smoothed_dark_ratio = None
        self.smoothed_bright_ratio = None
        self.current_brightness = 0
        self.brightness_error = 0
        self.state = ""
        self.mask = mask
        self.target_gain = 22 # optimal value in terms of efficiency
        self.exposure_min = 0
        self.exposure_max = 200000
        self.exposure_min_step = 50
        self.gain_min = 1.0
        self.gain_max = 40.0
        self.gain_min_step = 0.5
        self.target_brightness = 0.5
        self.compensation_factor = 0.62 # [0..1] amout of brightness_error to be compensated - to prevent from overshooting
        self.histogram_sampling = 512
        self.histogram_dark_point = 5
        self.histogram_bright_point = 507

    @staticmethod
    @njit
    def downsample_and_normalize(src):
        h, w = src.shape
        out = np.empty((h // 4, w // 4), dtype=np.float32)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                out[y // 4, x // 4] = src[y, x] / 65535.0
        return out

    def update(self, exposure_us, gain, ring_buffer: RingBuffer):
        #AE_start_time = time.perf_counter()
        gray_img = AEController.downsample_and_normalize(ring_buffer.frame_buffer[ring_buffer.frame_index])
        #if debug: print(f"compute for reading image from ring buffer and taking every 4th pixel in grey: {(time.perf_counter() - AE_start_time):.4f}s")

        # Histogram
        if self.mask is not None:
            mask_small = self.mask[::4, ::4]
            valid_mask = (mask_small > 0)
            histogram = cv2.calcHist([gray_img[valid_mask]], [0], None, [self.histogram_sampling], [0.0, 1.0]).flatten()
        else:
            histogram = cv2.calcHist([gray_img], [0], None, [self.histogram_sampling], [0.0, 1.0]).flatten()

        histogram /= histogram.sum()
        cdf = np.clip(np.cumsum(histogram), 0.0, 1.0)
        dark_ratio = cdf[self.histogram_dark_point]
        bright_ratio = 1.0 - cdf[self.histogram_bright_point]
        # if debug: print(f"AE: dark_ratio={dark_ratio:.8f}, bright_ratio={bright_ratio:.8f}")

        # Smoothing
        if self.smoothed_dark_ratio is None:
            self.smoothed_dark_ratio = dark_ratio
            self.smoothed_bright_ratio = bright_ratio
        else:
            self.smoothed_dark_ratio = self.SMOOTHING_ALPHA * dark_ratio + (1 - self.SMOOTHING_ALPHA) * self.smoothed_dark_ratio
            self.smoothed_bright_ratio = self.SMOOTHING_ALPHA * bright_ratio + (1 - self.SMOOTHING_ALPHA) * self.smoothed_bright_ratio

        self.current_brightness = 0.5 * (1.0 - self.smoothed_dark_ratio) + 0.5 * self.smoothed_bright_ratio

        if self.target_brightness > 0:
            self.brightness_error = np.round(np.log2(self.target_brightness / self.current_brightness), 2) # avoids jiggering
        else:
            self.brightness_error = 0.0

        self.state = ""
        adjusted_gain = gain

        # Gain
        if gain != self.target_gain:
            gain_error = self.compensation_factor * self.brightness_error
            proposed_gain = gain * np.power(2, gain_error)
            crosses_target = (gain - self.target_gain) * (proposed_gain - self.target_gain) < 0
            narrows_target = abs(proposed_gain - self.target_gain) < abs(gain - self.target_gain) and (gain - self.target_gain) * (proposed_gain - self.target_gain) >= 0

            if crosses_target:
                adjusted_gain = self.target_gain
                adjusted_gain = np.clip(adjusted_gain, self.gain_min, self.gain_max)
                self.state += " | setting gain to target"
            elif narrows_target:
                adjusted_gain = np.clip(proposed_gain, self.gain_min, self.gain_max)
                self.state += " | gain narrowing to sweet spot"

        gain_applied = np.log2(adjusted_gain / gain) if gain > 0 else 0.0
        remain_error = self.compensation_factor * self.brightness_error - gain_applied
        
        # Exposure control
        proposed_exposure = exposure_us * np.power(2, remain_error)

        if proposed_exposure < self.exposure_max:
            adjusted_exposure = proposed_exposure
            self.state += " | adjusting exposure"
        else:
            adjusted_exposure = self.exposure_max
            self.state += " | setting exposure to max"

        adjusted_exposure = np.clip(adjusted_exposure, self.exposure_min, self.exposure_max)

        exposure_applied = np.log2(adjusted_exposure / exposure_us) if exposure_us > 0 else 0.0
        leftover_error = remain_error - exposure_applied
        proposed_gain = adjusted_gain * np.power(2, leftover_error)
        adjusted_gain = np.clip(proposed_gain, self.gain_min, self.gain_max)
        self.state += " | adjusting gain for the rest"

        if abs(adjusted_exposure - exposure_us) >= self.exposure_min_step:
            new_exposure = adjusted_exposure
            self.state += " | exposure update"
        else:
            new_exposure = exposure_us
            self.state += " | no update for exposure"

        if abs(adjusted_gain - gain) >= self.gain_min_step:
            new_gain = adjusted_gain
            self.state += " | gain update"
        else:
            new_gain = gain
            self.state += " | no update for gain"

        if debug: print(f"AE(csv): {ring_buffer.frame_index},{self.current_brightness:.4f},{self.brightness_error:.4f},{int(new_exposure)},{new_gain:.2f},{self.state}")
        return new_exposure, new_gain

class QHYCameraController:
    # Constants
    ## SDK IDs
    CONTROL_EXPOSURE = 8
    CONTROL_GAIN = 6
    CONTROL_TRANSFERBIT = 10
    CONTROL_USBTRAFFIC = 12
    CONTROL_CURTEMP = 14
    CONTROL_CURPWM = 15
    CONTROL_MANULPWM = 16
    CONTROL_COOLER = 18
    
    ## FIXED Parameters
    RING_BUFFER_FRAMES = 200  # number of frames in ring buffer
    ROI_WIDTH = ROI_HEIGHT = 3200 # squared ROI, centered to zenith
    BITS_PER_PIXEL = 16  # image depth (bit-depth)
    COLOR_CHANNELS = 1  # Number of image color channels (monochrome)
    TARGET_TEMPERATURE = -20.0 # [°C]
    
    ## VARIABLE Parameters
    MASK_PATH = "../scripts/mask.png"
    TARGET_EXPOSURE = 70000 # [µs] initial exposure
    USB_SPEED = 6 # [0..60]
    FPS_TARGET = 5 # target FPS
    MIN_DELAY_SEC = 0.03 # [s] floor for initial delay estimation
    MAX_FRAME_GRABS = 10 # max number of successful frame grabs during calibration
    EXPOSURE_ADJUST_INTERVAL = 4 # frequency of exposure/gain adjustments

    ## Processing Parameters
    FRAME_GRAB_PENALTY_SEC = 1 / FPS_TARGET # forced sleep in case of failed frame grab [seconds]

    def __init__(self, sdk_path='/usr/local/lib/libqhyccd.so'):
        # Clean up leftover shared memory -------------------------------------------
        try:
            existing_shm = shared_memory.SharedMemory(name="cam_ring_buffer")
            existing_shm2 = shared_memory.SharedMemory(name="metadata_ring_buffer")
            existing_shm.close()
            existing_shm2.close()
            existing_shm.unlink()
            existing_shm2.unlink()
        except FileNotFoundError:
            pass
        except FileExistsError:
            pass

        self.bpp = c_uint32(self.BITS_PER_PIXEL)
        self.channels = c_uint32(self.COLOR_CHANNELS)
        self.ring_buffer = RingBuffer(self.ROI_WIDTH, self.ROI_HEIGHT)  # Initialize RingBuffer
        self.x_offset = int((5544 - self.ROI_WIDTH) / 2) # [px]
        self.y_offset = int((3684 - self.ROI_HEIGHT) / 2) # [px]
        self.cycle = 1 / self.FPS_TARGET # [s]
        self.delay = self.MIN_DELAY_SEC
        self.w = c_uint32(self.ROI_WIDTH)
        self.h = c_uint32(self.ROI_HEIGHT)
        self.successes = 0
        self.consecutive_failures = 0
        self.current_temp = 0.0
        self.target_temp = self.TARGET_TEMPERATURE
        self.cam_ready = True
        self.frame_bytes = self.ROI_WIDTH * self.ROI_HEIGHT
        self.temp_buffer = (ctypes.c_uint16 * self.frame_bytes)()

        # Reading the mask
        self.mask = cv2.imread(self.MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise RuntimeError(f"Could not load mask from {self.MASK_PATH}")

        # Auto-exposure
        self.ae_controller = AEController(mask=self.mask)

        # Loading SDK
        self.sdk = CDLL(sdk_path)
        self._set_function_signatures()

        # Initialize the camera
        self.cam = None
        self.initialize_camera()
        self.set_exposure_and_gain()

        # Calibration
        self.get_single_frame()
        self.set_live_mode()
        self.delay = self.calibrate_camera()
        if debug: print(f"Calibration: updated delay: {self.delay:.4f}")

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
        if debug: print(f"Initialisation: starting to initialize the camera")
        
        # Resetting USB
        try:
            subprocess.check_call(["usbreset", "Q183-Cool"])
        except subprocess.CalledProcessError as e:
            if debug: print("Resetting USB: ❌", e)
            exit(0)
        time.sleep(5.0)

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
        if (self.cam != None):
            if debug: print(f"Initialization: successfully opened camera {id_buffer.value.decode('utf-8').strip('\x00')} ✅")
        else:
            if debug: print(f"Initialization: failed to open camera ❌")
        time.sleep(0.5)

        # Setting 16-bit mode
        ret = self.sdk.SetQHYCCDBitsMode(self.cam, 16)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set bits mode to 16 ✅")
        else:
            if debug: print(f"Initialization: failed to set bits mode ❌")
        time.sleep(0.5)

        # Setting transfer bit
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT, 16)
        tbit = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set transfer bit to {tbit} ✅")
        else:
            if debug: print(f"Initialization: failed to set transfer bit ❌")
        time.sleep(0.5)

        # Setting resolution
        ret = self.sdk.SetQHYCCDResolution(
            self.cam,
            c_uint(self.x_offset),
            c_uint(self.y_offset),
            c_uint(self.ROI_WIDTH),
            c_uint(self.ROI_HEIGHT)
        )
        if (ret == 0):
            if debug: print(f"Initialization: successfully set resolution to {self.x_offset}, {self.y_offset}, {self.ROI_WIDTH},{self.ROI_HEIGHT} ✅")
        else:
            if debug: print(f"Initialization: failed to set resolution ❌")
        time.sleep(0.5)

        # Setting USB traffic
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_USBTRAFFIC, self.USB_SPEED)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set USB traffic to {self.USB_SPEED} ✅")
        else:
            if debug: print(f"Initialization: failed to set USB traffic ❌")
        time.sleep(0.5)

        # Initializing camera
        ret = self.sdk.InitQHYCCD(self.cam)
        if (ret == 0):
            if debug: print(f"Initialization: successfully initialized the camera ✅")
        else:
            if debug: print(f"Initialization: failed to initialize the camera ❌")
        time.sleep(2)

    def set_exposure_and_gain(self):
        if debug: print(f"Initialization: setting exposure and gain ...")
        # Setting exposure and gain
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE, self.TARGET_EXPOSURE)
        self.exposure = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set exposure to {self.exposure} ✅")
        else:
            if debug: print(f"Initialization: failed to set exposure ❌")

        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.ae_controller.target_gain)
        self.gain = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_GAIN)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set gain to {self.gain} ✅")
        else:
            if debug: print(f"Initialization: failed to set gain ❌")
        time.sleep(0.5)

    def set_live_mode(self):
        if debug: print(f"Initialisation: setting live mode ...")
        # Setting mode for live frames
        ret = self.sdk.SetQHYCCDStreamMode(self.cam, 1)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set to live mode ✅")
        else:
            if debug: print(f"Initialization: failed to set live mode ❌")
        time.sleep(0.5)

        # Setting resolution
        ret = self.sdk.SetQHYCCDResolution(self.cam, c_uint(self.x_offset), c_uint(self.y_offset), c_uint(self.ROI_WIDTH), c_uint(self.ROI_HEIGHT))
        if (ret == 0):
            if debug: print(f"Initialization: successfully set resolution to {self.x_offset}, {self.y_offset}, {self.ROI_WIDTH},{self.ROI_HEIGHT} ✅")
        else:
            if debug: print(f"Initialization: failed to set resolution ❌")
        time.sleep(0.5)

        # Begin live
        ret = self.sdk.BeginQHYCCDLive(self.cam)
        if (ret == 0):
            if debug: print(f"Initialization: successfully began live mode ✅")
        else:
            if debug: print(f"Initialization: failed to begin live mode ❌")
        time.sleep(0.5)

    def get_single_frame(self):
        if debug: print(f"Calibration: fetching 3 single frames ...")
        # Temporary buffer for single frame retrieval
        #self.frame_bytes = self.ROI_WIDTH * self.ROI_HEIGHT
        #self.temp_buffer = (ctypes.c_uint16 * self.frame_bytes)()
        for i in range(3):
            if debug: print(f"Calibration: exposing a single frame ...")
            ret = self.sdk.ExpQHYCCDSingleFrame(self.cam)
            if (ret == 0):
                if debug: print(f"Calibration: successfully exposed a single frame ✅")
            else:
                if debug: print(f"Calibration: failed to expose a single frame ❌")
                continue
            time.sleep(0.5)
            ret = self.sdk.GetQHYCCDSingleFrame(self.cam, byref(self.w), byref(self.h),
                                                byref(self.bpp), byref(self.channels), self.temp_buffer)
            if (ret == 0):
                if debug: print(f"Calibration: successfully fetched a single frame ✅")
            else:
                if debug: print(f"Calibration: failed to fetch a single frame ❌")
            time.sleep(1.5)
        return

    def calibrate_camera(self):
        if debug: print(f"Calibration: fetching 10 live frames ...")
        self.delay = self.MIN_DELAY_SEC
        self.successes = 0
        consecutive_failures = 0
        ret = -1

        while self.successes < self.MAX_FRAME_GRABS:
            loop_start = time.perf_counter()

            # Try fetching a frame
            ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.temp_buffer)

            if ret == 0:
                consecutive_failures = 0
                img = np.frombuffer(self.temp_buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
                self.successes += 1
                
                if debug: print(f"Calibration: successfully fetched a live frame, successes={self.successes} ✅")
            else:
                consecutive_failures += 1
                time.sleep(abs(self.FRAME_GRAB_PENALTY_SEC - self.delay))

                if consecutive_failures >= 5:
                    if debug: print(f"Calibration: 5 consecutive live frame grabs ❌")
                    time.sleep(abs(self.FRAME_GRAB_PENALTY_SEC - self.delay) * 5)
                    consecutive_failures = 0

                self.successes -= 1
                if debug: print(f"Calibration: failed to fetch a live frame ❌")
                ret = -1

            time.sleep(self.delay)
            
        return self.delay

    # Temperature control
    def control_temperature_pwm(self):
        self.current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
        self.current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)
        if debug: print(f"Cooling: T={self.current_temp}°C, PWM={(100 * self.current_pwm / 255):.1f}%")

        if self.current_temp < -100:
            if debug: print("Warning: Invalid temperature reading, retrying...")
            time.sleep(1.0)  # Blocking, consider logging instead
            self.current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
            self.current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)

        if self.current_temp >= self.target_temp:
            self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_COOLER, self.target_temp)

    def get_live_frame(self):
        self.cam_ready = False
        #print(f"CAM: cam_ready is {self.cam_ready}")
        prev_time = time.perf_counter()
        ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.temp_buffer)
        
        # Retry in case of unsuccessful data aquisition
        if (ret != 0):
            time.sleep(abs(self.FRAME_GRAB_PENALTY_SEC - self.delay)) # sleep before aquiring again
            if debug: print("retrying frame aquisition ...")
            ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.temp_buffer)
        
        if (ret == 0):
            # Convert buffer to image
            img = np.frombuffer(self.temp_buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))

            # Apply binary mask
            if self.mask is not None:
                img[self.mask == 0] = 0

            # Storing the data
            self.ring_buffer.store_frame(img, self.exposure, self.gain)

            # Auto-exposure correction at intervals
            if (self.ring_buffer.frame_index % self.EXPOSURE_ADJUST_INTERVAL) == 0:         # and self.ring_buffer.frame_index != 0 ... to prevent AE control on the first frame
                prev_exposure, prev_gain = self.exposure, self.gain

                # Compute new exposure and gain using the ring buffer
                self.exposure, self.gain = self.ae_controller.update(self.exposure, self.gain, self.ring_buffer)

                # Apply changes if significant
                if abs(self.exposure - prev_exposure) > 0:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE, self.exposure)

                if abs(self.gain - prev_gain) > 0:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.gain)

            # Perform temperature control at intervals
            if (self.ring_buffer.frame_index == 11) or (self.ring_buffer.frame_index == 111):
                self.control_temperature_pwm()

            self.ring_buffer.frame_index = (self.ring_buffer.frame_index + 1) % self.ring_buffer.buffer_size  # Cycle index
            
            # Delay estimation before the next frame is aquired to keep target_FPS
            processing_time = time.perf_counter() - prev_time
            self.delay = 1 / self.FPS_TARGET - processing_time
            if self.delay < 0: self.delay = 0
            estimated_fps = 1 / (processing_time + self.delay)
            if debug: print(f"Live: Index={self.ring_buffer.frame_index}, Processing time={processing_time:.4f}s, Delay={self.delay:.4f}s, Estimated FPS={estimated_fps:.1f} ✅")
            
            time.sleep(self.delay)
            self.cam_ready = True
            #print(f"CAM: cam_ready is {self.cam_ready}")
            return img  # Return the processed frame
            
        return None


    def get_shape(self):
        return (self.h.value, self.w.value)


    def get_exposure(self):
        return self.exposure


    def get_gain(self):
        return self.gain


    def get_temperature(self):
        #return self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP) # commented, because it calls too often, while current_temp is updated every 100 frames only
        return self.current_temp


    def set_target_brightness(self, target_brightness):
        self.ae_controller.target_brightness = target_brightness


    def set_target_gain(self, target_gain):
        self.ae_controller.target_gain = target_gain


    def set_exposure_min(self, exposure_min):
        self.ae_controller.exposure_min = exposure_min


    def set_exposure_max(self, exposure_max):
        self.ae_controller.exposure_max = exposure_max


    def set_gain_min(self, gain_min):
        self.ae_controller.gain_min = gain_min


    def set_gain_max(self, gain_max):
        self.ae_controller.gain_max = gain_max


    def set_exposure_min_step(self, exposure_min_step):
        self.ae_controller.exposure_min_step = exposure_min_step


    def set_gain_min_step(self, gain_min_step):
        self.ae_controller.gain_min_step = gain_min_step


    def set_compensation_factor(self, compensation_factor):
        self.ae_controller.compensation_factor = compensation_factor


    def set_target_temperature(self, target_temperature):
        self.target_temp = target_temperature
    
    def set_histogram_sampling(self, histogram_sampling):
        self.histogram_sampling = histogram_sampling
    
    def set_histogram_dark_point(self, histogram_dark_point):
        self.histogram_dark_point = histogram_dark_point
    
    def set_histogram_bright_point(self, histogram_bright_point):
        self.histogram_bright_point = histogram_bright_point

    def close(self):
        # Stop live camera
        try:
            self.sdk.StopQHYCCDLive(self.cam)
            time.sleep(0.5)
        except Exception as e:
            if debug: print(f"Error stopping live camera: {e}")

        # Stop cooler
        try:
            self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_MANULPWM, 0)
            time.sleep(0.5)
        except Exception as e:
            if debug: print(f"Error stopping cooler: {e}")

        # Cleaning up shared memory buffers
        self.ring_buffer.cleanup()

        # Release SDK resources
        self.sdk.ReleaseQHYCCDResource()
        time.sleep(0.2)

        # Close camera connection
        self.sdk.CloseQHYCCD(self.cam)
        time.sleep(2.0)

        #cv2.destroyAllWindows()
        time.sleep(0.2)

        # Reset USB
        try:
            subprocess.check_call(["usbreset", "Q183-Cool"])
        except subprocess.CalledProcessError as e:
            if debug: print("Resetting USB: ❌", e)
        time.sleep(5.0)


if __name__ == "__main__":

    qhy_camera = QHYCameraController()

    print(f"Live: fetching live frames ...")

    # Live loop
    while True:
        #prev_time = time.perf_counter()
        img = qhy_camera.get_live_frame()

        if img is None:
            print(f"Live: Index={qhy_camera.ring_buffer.frame_index + 1} not acquired ❌ ... sleeping for {qhy_camera.delay:.4f}s")
            time.sleep(qhy_camera.delay)

        if quit_requested:
            break
        
    # Close
    qhy_camera.close()

    exit(0)
