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
    TARGET_GAIN = 22
    EXPOSURE_MIN = 0
    EXPOSURE_MAX = 140000
    GAIN_MIN = 1.0
    GAIN_MAX = 54.0
    EXPOSURE_MIN_STEP = 100
    GAIN_MIN_STEP = 0.5
    TARGET_BRIGHTNESS = 0.5
    COMPENSATION_FACTOR = 0.62 # [0..1] amout of brightness_error to be compensated - to prevent from overshooting
    SMOOTHING_ALPHA = 0.3

    def __init__(self):
        self.smoothed_dark_ratio = None
        self.smoothed_bright_ratio = None
        self.current_brightness = 0
        self.brightness_error = 0
        self.state = ""

    def update(self, exposure_us, gain, ring_buffer: RingBuffer):
        gray_img = (ring_buffer.frame_buffer[ring_buffer.frame_index] / 65535.0).astype(np.float32)[::4, ::4]
        histogram = cv2.calcHist([gray_img], [0], None, [512], [0.0, 1.0]).flatten()
        histogram /= histogram.sum()
        cdf = np.clip(np.cumsum(histogram), 0.0, 1.0)
        dark_ratio = cdf[5]
        bright_ratio = 1.0 - cdf[507]
        if debug: print(f"AE: dark_ratio={dark_ratio:.8f}, bright_ratio={bright_ratio:.8f}")

        if self.smoothed_dark_ratio is None:
            self.smoothed_dark_ratio = dark_ratio
            self.smoothed_bright_ratio = bright_ratio
        else:
            self.smoothed_dark_ratio = self.SMOOTHING_ALPHA * dark_ratio + (1 - self.SMOOTHING_ALPHA) * self.smoothed_dark_ratio
            self.smoothed_bright_ratio = self.SMOOTHING_ALPHA * bright_ratio + (1 - self.SMOOTHING_ALPHA) * self.smoothed_bright_ratio

        self.current_brightness = 0.5 * (1.0 - self.smoothed_dark_ratio) + 0.5 * self.smoothed_bright_ratio

        if self.TARGET_BRIGHTNESS > 0:
            self.brightness_error = np.round(np.log2(self.TARGET_BRIGHTNESS / self.current_brightness), 2) # avoids jiggering
        else:
            self.brightness_error = 0.0

        self.state = ""
        adjusted_gain = gain

        if gain != self.TARGET_GAIN:
            gain_error = self.COMPENSATION_FACTOR * self.brightness_error
            proposed_gain = gain * np.power(2, gain_error)
            crosses_target = (gain - self.TARGET_GAIN) * (proposed_gain - self.TARGET_GAIN) < 0
            narrows_target = abs(proposed_gain - self.TARGET_GAIN) < abs(gain - self.TARGET_GAIN) and (gain - self.TARGET_GAIN) * (proposed_gain - self.TARGET_GAIN) >= 0

            if crosses_target:
                adjusted_gain = self.TARGET_GAIN
                adjusted_gain = np.clip(adjusted_gain, self.GAIN_MIN, self.GAIN_MAX)
                self.state += " | setting gain to target"
            elif narrows_target:
                adjusted_gain = np.clip(proposed_gain, self.GAIN_MIN, self.GAIN_MAX)
                self.state += " | gain narrowing to sweet spot"

        gain_applied = np.log2(adjusted_gain / gain) if gain > 0 else 0.0
        remain_error = self.COMPENSATION_FACTOR * self.brightness_error - gain_applied
        proposed_exposure = exposure_us * np.power(2, remain_error)

        if proposed_exposure < self.EXPOSURE_MAX:
            adjusted_exposure = proposed_exposure
            self.state += " | adjusting exposure"
        else:
            adjusted_exposure = self.EXPOSURE_MAX
            self.state += " | setting exposure to max"

        adjusted_exposure = np.clip(adjusted_exposure, self.EXPOSURE_MIN, self.EXPOSURE_MAX)

        exposure_applied = np.log2(adjusted_exposure / exposure_us) if exposure_us > 0 else 0.0
        leftover_error = remain_error - exposure_applied
        proposed_gain = adjusted_gain * np.power(2, leftover_error)
        adjusted_gain = np.clip(proposed_gain, self.GAIN_MIN, self.GAIN_MAX)
        self.state += " | adjusting gain for the rest"

        if abs(adjusted_exposure - exposure_us) >= self.EXPOSURE_MIN_STEP:
            new_exposure = adjusted_exposure
            self.state += " | exposure update"
        else:
            new_exposure = exposure_us
            self.state += " | no update for exposure"

        if abs(adjusted_gain - gain) >= self.GAIN_MIN_STEP:
            new_gain = adjusted_gain
            self.state += " | gain update"
        else:
            new_gain = gain
            self.state += " | no update for gain"

        if debug: print(f"AE(csv): {ring_buffer.frame_index},{self.current_brightness:.4f},{self.brightness_error:.4f},{int(new_exposure)},{new_gain:.2f},{self.state}")
        return new_exposure, new_gain

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
    MAX_FRAME_GRABS = 10 # max number of successful frame grabs during calibration
    EXPOSURE_ADJUST_INTERVAL = 4 # Frequency of exposure/gain adjustments

    # Buffer & Processing Parameters
    FRAME_SYNC_DELAY_STEP = -0.001  # Decreasing delay step with new frame grabs [seconds]
    FRAME_GRAB_PENALTY_SEC = 0.31  # Forced sleep in case of failed frame grab [seconds]

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
        self.delay = max(self.MIN_DELAY_SEC, self.cycle - self.MIN_GRAB_TIME_SEC) # [s]
        self.w = c_uint32(self.ROI_WIDTH)
        self.h = c_uint32(self.ROI_HEIGHT)
        self.successes = 0
        self.consecutive_failures = 0

        # Auto-exposure
        self.ae_controller = AEController()

        # Loading SDK
        self.sdk = CDLL(sdk_path)
        self._set_function_signatures()
        
        # Initialize the camera
        self.cam = None
        self.initialize_camera()
        self.set_exposure_and_gain()
        
        # CV2 window
        #if debug:
        #    window_title = "Sky360 debug preview - <q> = quit"
        #    cv2.namedWindow(window_title, cv2.WINDOW_OPENGL)
        #    cv2.resizeWindow(window_title, 800, 800)
        #    print(f"Initialization: successfully opened CV2 window ✅")

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
        time.sleep(0.2)

        # Setting 16-bit mode
        ret = self.sdk.SetQHYCCDBitsMode(self.cam, 16)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set bits mode to 16 ✅")
        else:
            if debug: print(f"Initialization: failed to set bits mode ❌")
        time.sleep(0.2)

        # Setting transfer bit
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT, 16)
        tbit = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_TRANSFERBIT)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set transfer bit to {tbit} ✅")
        else:
            if debug: print(f"Initialization: failed to set transfer bit ❌")
        time.sleep(0.2)

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
        time.sleep(0.2)

        # Setting USB traffic to 0
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_USBTRAFFIC, 0)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set USB traffic to automatic (0) ✅")
        else:
            if debug: print(f"Initialization: failed to set USB traffic ❌")
        time.sleep(0.2)

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
        
        ret = self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.TARGET_GAIN)
        self.gain = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_GAIN)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set gain to {self.gain} ✅")
        else:
            if debug: print(f"Initialization: failed to set gain ❌")
        time.sleep(0.2)

    def set_live_mode(self):
        if debug: print(f"Initialisation: setting live mode ...")
        # Setting mode for live frames
        ret = self.sdk.SetQHYCCDStreamMode(self.cam, 1)
        if (ret == 0):
            if debug: print(f"Initialization: successfully set to live mode ✅")
        else:
            if debug: print(f"Initialization: failed to set live mode ❌")
        time.sleep(0.2)

        # Setting resolution
        ret = self.sdk.SetQHYCCDResolution(self.cam, c_uint(self.x_offset), c_uint(self.y_offset), c_uint(self.ROI_WIDTH), c_uint(self.ROI_HEIGHT))
        if (ret == 0):
            if debug: print(f"Initialization: successfully set resolution to {self.x_offset}, {self.y_offset}, {self.ROI_WIDTH},{self.ROI_HEIGHT} ✅")
        else:
            if debug: print(f"Initialization: failed to set resolution ❌")
        time.sleep(0.2)

        # Begin live
        ret = self.sdk.BeginQHYCCDLive(self.cam)
        if (ret == 0):
            if debug: print(f"Initialization: successfully began live mode ✅")
        else:
            if debug: print(f"Initialization: failed to begin live mode ❌")
        time.sleep(0.2)

    def get_single_frame(self):
        if debug: print(f"Calibration: fetching 3 single frames ...")
        # Temporary buffer for single frame retrieval
        self.frame_bytes = self.ROI_WIDTH * self.ROI_HEIGHT
        self.temp_buffer = (ctypes.c_uint16 * self.frame_bytes)()
        for i in range(3):
            if debug: print(f"Calibration: exposing a single frame ...")
            ret = self.sdk.ExpQHYCCDSingleFrame(self.cam)
            if (ret == 0):
                if debug: print(f"Calibration: successfully exposed a single frame ✅")
            else:
                if debug: print(f"Calibration: failed to expose a single frame ❌")
            time.sleep(0.1)
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
        delay = self.MIN_DELAY_SEC
        self.successes = 0
        consecutive_failures = 0
        ret = -1

        while self.successes < self.MAX_FRAME_GRABS:
            loop_start = time.perf_counter()

            # Try fetching a frame
            ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.temp_buffer)
            grab_time = time.perf_counter() - loop_start

            if ret == 0:
                consecutive_failures = 0
                img_start = time.perf_counter()
                img = np.frombuffer(self.temp_buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
                processing_time = time.perf_counter() - img_start
                duration = grab_time + processing_time
                self.speed = self.frame_bytes / duration
                self.successes += 1
                self.estimated_fps = 1.0 / duration

                # Update delay dynamically
                grab_and_process_time = time.perf_counter() - loop_start
                delay = max(self.MIN_DELAY_SEC, self.cycle - grab_and_process_time)
                if debug: print(f"Calibration: successfully fetched a live frame, delay={delay:.4f}s, successes={self.successes} ✅")
            else:
                consecutive_failures += 1
                time.sleep(self.FRAME_GRAB_PENALTY_SEC)

                if consecutive_failures >= 5:
                    if debug: print(f"Calibration: 5 consecutive live frame grabs ❌")
                    time.sleep(self.FRAME_GRAB_PENALTY_SEC * 5)

                    # Stop & restart live camera to resync
                    ret = self.sdk.StopQHYCCDLive(self.cam)
                    if (ret == 0):
                        if debug: print(f"Calibration: successfully stopped live mode ✅")
                    else:
                        if debug: print(f"Calibration: failed to stop live mode ❌")
                    time.sleep(0.2)
                    
                    ret = self.sdk.BeginQHYCCDLive(self.cam)
                    if (ret == 0):
                        if debug: print(f"Calibration: successfully began live mode ✅")
                    else:
                        if debug: print(f"Calibration: failed to begin live mode ❌")
                    time.sleep(0.2)

                    consecutive_failures = 0

                delay += self.FRAME_SYNC_DELAY_STEP
                delay = max(self.MIN_DELAY_SEC, min(delay, self.cycle))
                self.FRAME_SYNC_DELAY_STEP = self.FRAME_SYNC_DELAY_STEP if delay not in (self.MIN_DELAY_SEC, self.cycle) else -self.FRAME_SYNC_DELAY_STEP
                self.successes -= 1
                if debug: print(f"Calibration: failed to fetch a live frame ❌")
                ret = -1
                
            time.sleep(delay)
            
        return delay

    # Temperature control
    def control_temperature_pwm(self):
        current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
        current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)
        if debug: print(f"Cooling: T={current_temp}°C, PWM={(100 * current_pwm / 255):.1f}%")

        if current_temp < -100:
            if debug: print("Warning: Invalid temperature reading, retrying...")
            time.sleep(1.0)  # Blocking, consider logging instead
            current_temp = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURTEMP)
            current_pwm = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_CURPWM)

        if abs(current_temp - self.TARGET_TEMP) > self.TEMPERATURE_TOLERANCE:
            self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_COOLER, self.TARGET_TEMP)

    def get_live_frame(self):
        ret = self.sdk.GetQHYCCDLiveFrame(self.cam, ctypes.byref(self.w), ctypes.byref(self.h), ctypes.byref(self.bpp), ctypes.byref(self.channels), self.temp_buffer)

        if (ret == 0):
            # Convert buffer to image
            img = np.frombuffer(self.temp_buffer, dtype=np.uint16).reshape((self.h.value, self.w.value))
            self.ring_buffer.store_frame(img, self.exposure, self.gain)

            # Auto-exposure correction at intervals
            if (self.ring_buffer.frame_index % self.EXPOSURE_ADJUST_INTERVAL) == 0:
                prev_exposure, prev_gain = self.exposure, self.gain

                # Compute new exposure and gain using the ring buffer
                self.exposure, self.gain = self.ae_controller.update(self.exposure,
                                                                     self.gain,
                                                                     self.ring_buffer)

                # Apply changes if significant
                if abs(self.exposure - prev_exposure) > 0:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE, self.exposure)
                    #self.exposure = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_EXPOSURE)

                if abs(self.gain - prev_gain) > 0:
                    self.sdk.SetQHYCCDParam(self.cam, self.CONTROL_GAIN, self.gain)
                    #self.gain = self.sdk.GetQHYCCDParam(self.cam, self.CONTROL_GAIN)

            # Perform temperature control at intervals
            if (self.ring_buffer.frame_index == 11) or (self.ring_buffer.frame_index == 111):
                self.control_temperature_pwm()
            
            self.ring_buffer.frame_index = (self.ring_buffer.frame_index + 1) % self.ring_buffer.buffer_size  # Cycle index
            return img  # Return the processed frame
        return None

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

    if debug: print(f"Live: fetching live frames ...")
    
    # Live loop
    while True:
        prev_time = time.perf_counter()
        img = qhy_camera.get_live_frame()

        if img is not None and debug:
            fps = 1 / (time.perf_counter() - prev_time)
            qhy_camera.ring_buffer.frame_index
            if debug: print(f"Live: Index={qhy_camera.ring_buffer.frame_index}, FPS={fps:.1f} ✅")
            
            # optional: for viewing (downsampled)
            #debayered_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
            #scaled_img = debayered_img[::4, ::4]
            #cv2.imshow("Image", scaled_img)

        if quit_requested:
            break
        
        time.sleep(qhy_camera.delay)

    # Close
    qhy_camera.close()

    exit(0)
