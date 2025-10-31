import sys
import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
import multiprocessing
import subprocess
import os
from datetime import datetime, timezone

from ecal_lib.ecal_util.proto_receiver import ProtoReceiver
from ecal_lib.ecal_util.proto_sender import ProtoSender
from ecal_lib.ecal_util.image_message import ImageMessage
from ecal_lib.ecal_util.set_process_name import *

QHY_CHANNEL_NAME = "QHYCamera"
#MESSAGE_NAME = "web_camera_image"
#PROTO_FILE = "ecal_lib.proto_files.web_camera_image_pb2"
MESSAGE_NAME = "qhy_camera_image"
PROTO_FILE = "ecal_lib.proto_files.qhy_camera_image_pb2"

STATUS_MESSAGE_NAME = "qhy_camera_status"
STATUS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_status_pb2"
QHY_STATUS_CHANNEL = "QHYCamera_status"

PARAMS_MESSAGE_NAME = "qhy_camera_parameters"
PARAMS_PROTO_FILE = "ecal_lib.proto_files.qhy_camera_parameters_pb2"
QHY_PARAMS_CHANNEL = "QHYCamera_parameters"

DEFAULT_TARGET_BRIGHTNESS = 0.5
DEFAULT_TARGET_GAIN = 22
DEFAULT_EXPOSURE_MIN = 0
DEFAULT_EXPOSURE_MAX = 200000
DEFAULT_GAIN_MIN = 1.0
DEFAULT_GAIN_MAX = 30.0
DEFAULT_EXPOSURE_MIN_STEP = 1
DEFAULT_GAIN_MIN_STEP = 0.1
DEFAULT_COMPENSATION_FACTOR = 0.62
DEFAULT_TARGET_TEMPERATURE = -20.0 # [°C]
DEFAULT_HISTOGRAM_SAMPLING = 512
DEFAULT_HISTOGRAM_DARK_POINT = 5
DEFAULT_HISTOGRAM_BRIGHT_POINT = 507

"""
def view_ecal_video():
    # Set process name
    set_process_name(f"eCAL viewer on {QHY_CHANNEL_NAME}")

    proto_rec = ProtoReceiver(QHY_CHANNEL_NAME, MESSAGE_NAME, PROTO_FILE)
    image_msg = ImageMessage(MESSAGE_NAME)

    cv2.namedWindow(QHY_CHANNEL_NAME + " Viewer", cv2.WINDOW_NORMAL)

    while True:
        received, proto_rec_message, _ = proto_rec.receive(100)
        if received:
            image_msg.process_message(proto_rec_message)
            frame = image_msg.get_rgb_image()

            # Display the image
            cv2.imshow(QHY_CHANNEL_NAME + " Viewer", frame)

        k = cv2.waitKey(1)
        # Esc key to stop
        if k == 27:
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()
"""

class CameraMonitorApp:

    def __init__(self, root):
        # Initialize eCAL
        set_process_name(f"{QHY_CHANNEL_NAME} Monitor App")

        # Proto sender
        self.params_proto_snd = ProtoSender(QHY_PARAMS_CHANNEL, PARAMS_MESSAGE_NAME, PARAMS_PROTO_FILE)

        # Proto receiver
        self.status_proto_rec = ProtoReceiver(QHY_STATUS_CHANNEL, STATUS_MESSAGE_NAME, STATUS_PROTO_FILE)

        # set multiprocessing to spawn
        multiprocessing.set_start_method("spawn")

        self.root = root
        root.title("QHY Camera Monitor")

        # Create left, right and message frames
        left_frame = tk.Frame(root)
        left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.left_frame = left_frame

        right_frame = tk.Frame(root, bg='#3399FF')
        right_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.right_frame = right_frame

        msg_frame = tk.Frame(root, bg='#333')
        msg_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.msg_frame = msg_frame

        msg_frame.grid_columnconfigure(1, weight=1)


        # UI elements
        self.date_var = tk.StringVar(value="0000-00-00")
        self.time_var = tk.StringVar(value="00:00:00.000 UTC")
        self.temp_var = tk.DoubleVar(value=0.0)
        self.gain_var = tk.DoubleVar(value=0.0)
        self.exposure_var = tk.DoubleVar(value=0.0)

        self.target_brightness_var = tk.DoubleVar(value=DEFAULT_TARGET_BRIGHTNESS)
        self.target_gain_var = tk.DoubleVar(value=DEFAULT_TARGET_GAIN)
        self.exposure_min_var = tk.DoubleVar(value=DEFAULT_EXPOSURE_MIN)
        self.exposure_max_var = tk.DoubleVar(value=DEFAULT_EXPOSURE_MAX)
        self.gain_min_var = tk.DoubleVar(value=DEFAULT_GAIN_MIN)
        self.gain_max_var = tk.DoubleVar(value=DEFAULT_GAIN_MAX)
        self.exposure_min_step_var = tk.DoubleVar(value=DEFAULT_EXPOSURE_MIN_STEP)
        self.gain_min_step_var = tk.DoubleVar(value=DEFAULT_GAIN_MIN_STEP)
        self.compensation_factor_var = tk.DoubleVar(value=DEFAULT_COMPENSATION_FACTOR)
        self.target_temperature_var  = tk.DoubleVar(value=DEFAULT_TARGET_TEMPERATURE)
        self.histogram_sampling_var  = tk.DoubleVar(value=DEFAULT_HISTOGRAM_SAMPLING)
        self.histogram_dark_point_var  = tk.DoubleVar(value=DEFAULT_HISTOGRAM_DARK_POINT)
        self.histogram_bright_point_var  = tk.DoubleVar(value=DEFAULT_HISTOGRAM_BRIGHT_POINT)

        self.support_msg = tk.StringVar(value="Status: nominal")
        self.camera_msg = tk.StringVar(value="")

        ##### UI

        ### Left Frame

        # Row 0
        ttk.Label(left_frame, text="Timestamp:").grid(row=0, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
        ttk.Label(left_frame, textvariable=self.date_var).grid(row=0, column=1, padx=5, pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 1
        ttk.Label(left_frame, textvariable=self.time_var).grid(row=1, column=1, padx=5, pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 2
        ttk.Label(left_frame, text="Temperature [°C]:").grid(row=2, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Label(left_frame, textvariable=self.temp_var).grid(row=2, column=1, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 3
        ttk.Label(left_frame, text="Gain [dB]:").grid(row=3, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Label(left_frame, textvariable=self.gain_var).grid(row=3, column=1, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 4
        ttk.Label(left_frame, text="Exposure [µs]:").grid(row=4, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Label(left_frame, textvariable=self.exposure_var).grid(row=4, column=1, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 5
        ttk.Label(left_frame, text="").grid(row=5, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')   
        
        # Row 6        
        ttk.Button(left_frame, text="Close QHY", command=self.close_qhy, width=15).grid(row=6, column=0)
        ttk.Button(left_frame, text="Reset QHY", command=self.reset_qhy, width=15).grid(row=6, column=1)
        
        # Row 7
        self.view_button = ttk.Button(left_frame, text="View QHY Image", command=self.view_qhy_image, width=15)
        self.view_button.grid(row=7, column=0)
        self.mask_editor_button = ttk.Button(left_frame, text="Mask Editor", command=self.edit_mask, width=15)
        self.mask_editor_button.grid(row=7, column=1)
        
        ### Right Frame
        
        # Row 0
        ttk.Label(right_frame, text="target_brightness:", background='#3399FF').grid(row=0, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.target_brightness_var, width=6).grid(row=0, column=1, sticky='w')
        ttk.Label(right_frame, text="target_temperature:", background='#3399FF').grid(row=0, column=2, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.target_temperature_var, width=6).grid(row=0, column=3, sticky='w')
        
        # Row 1
        ttk.Label(right_frame, text="target_gain:", background='#3399FF').grid(row=1, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.target_gain_var, width=6).grid(row=1, column=1, sticky='w')
        ttk.Label(right_frame, text="compensation_factor:", background='#3399FF').grid(row=1, column=2, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.compensation_factor_var, width=6).grid(row=1, column=3, sticky='w')
        
        # Row 2
        ttk.Label(right_frame, text="exposure_min:", background='#3399FF').grid(row=2, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.exposure_min_var, width=6).grid(row=2, column=1, sticky='w')
        ttk.Label(right_frame, text="exposure_max:", background='#3399FF').grid(row=2, column=2, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.exposure_max_var, width=6).grid(row=2, column=3, sticky='w')
        ttk.Label(right_frame, text="exposure_min_step:", background='#3399FF').grid(row=2, column=4, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.exposure_min_step_var, width=6).grid(row=2, column=5, sticky='w')
        
        # Row 3
        ttk.Label(right_frame, text="gain_min:", background='#3399FF').grid(row=3, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.gain_min_var, width=6).grid(row=3, column=1, sticky='w')
        ttk.Label(right_frame, text="gain_max:", background='#3399FF').grid(row=3, column=2, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.gain_max_var, width=6).grid(row=3, column=3, sticky='w')
        ttk.Label(right_frame, text="gain_min_step:", background='#3399FF').grid(row=3, column=4, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.gain_min_step_var, width=6).grid(row=3, column=5, sticky='w')
        
        # Row 4
        ttk.Label(right_frame, text="histogram_sampling:", background='#3399FF').grid(row=4, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.histogram_sampling_var, width=6).grid(row=4, column=1, sticky='w')
        ttk.Label(right_frame, text="histogram_dark_point:", background='#3399FF').grid(row=4, column=2, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.histogram_dark_point_var, width=6).grid(row=4, column=3, sticky='w')
        ttk.Label(right_frame, text="histogram_bright_point:", background='#3399FF').grid(row=4, column=4, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        ttk.Entry(right_frame, textvariable=self.histogram_bright_point_var, width=6).grid(row=4, column=5, sticky='w')
        
        # Row 5
        ttk.Label(right_frame, text="", background='#3399FF').grid(row=5, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')

        # Row 6
        ttk.Label(right_frame, text="", background='#3399FF').grid(row=6, column=0, padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
        
        # Row 7
        ttk.Button(right_frame, text="Apply Parameters", command=self.apply_parameters, width=15).grid(row=7, column=2)
        
        ### Message Frame
        
        # Row 0
        ttk.Label(msg_frame, textvariable=self.support_msg, background='#333', foreground='#FFF').grid(row=0, column=0)
        ttk.Label(msg_frame, textvariable=self.camera_msg, background='#600', foreground='#FFF').grid(row=0, column=1)


        # Start status polling
        self.running = True
        threading.Thread(target=self.status_listener, daemon=True).start()
        self.view_process = None
        self.mask_edit_process = None
        # Poll process status
        self.check_process_status()


    def status_listener(self):
        while self.running:
            status_proto_received, status_msg, _ = self.status_proto_rec.receive(100)
            if status_proto_received:
                self.temp_var.set(f"{status_msg.temperature:.1f}") # reduced to 1 digit for readability
                self.gain_var.set(f"{status_msg.gain:.2f}") # reduced to 2 digits for readability
                self.exposure_var.set(int(status_msg.exposure)) # reduced to 0 digits for readability
                if status_msg.is_qhy_live:
                    self.camera_msg.set("QHY is running")
                else:
                    self.camera_msg.set("QHY crashed")
                
                # Convert UNIX timestamp (float or int) → formatted UTC strings
                ts = status_msg.time_stamp / 1e6
                dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
                self.date_var.set(dt_utc.strftime("%Y-%m-%d"))
                self.time_var.set(dt_utc.strftime("%H:%M:%S.%f")[:-3] + " UTC")
                
            time.sleep(0.1)


    def reset_qhy(self):
        self.camera_msg.set("Resetting QHY")
        params_message = self.params_proto_snd.get_message_type()
        params_message.close_qhy = False
        params_message.reset_qhy = True
        self.params_proto_snd.send(params_message)


    def close_qhy(self):
        self.camera_msg.set("Closing QHY")
        params_message = self.params_proto_snd.get_message_type()
        params_message.close_qhy = True
        params_message.reset_qhy = False
        self.params_proto_snd.send(params_message)


    def get_parameters(self, param, param_name, param_default):
        try:
            new_param = param.get()
        except Exception as err:
            self.support_msg.set(f"{param_name}: {err}")
            param.set(param_default)
            new_param = param_default
        return new_param
    

    def view_qhy_image(self):
        if self.view_process and self.view_process.poll() is None:
            self.support_msg.set("Viewer already running.")
            return
        self.view_button.state(['disabled'])
        try:
            viewer_script = os.path.join(os.path.dirname(__file__), "../scripts/viewer_qhy_camera.py")
            self.view_process = subprocess.Popen([sys.executable, viewer_script])
        except Exception as e:
            self.support_msg.set(f"Failed to start viewer: {e}")
            self.view_button.state(['!disabled'])


    def edit_mask(self):
        if self.mask_edit_process and self.mask_edit_process.poll() is None:
            self.support_msg.set("Mask Editor already running.")
            return
        self.mask_editor_button.state(['disabled'])
        try:
            mask_editor_script = os.path.join(os.path.dirname(__file__), "../scripts/mask_editor.py")
            self.mask_edit_process = subprocess.Popen([sys.executable, mask_editor_script])
        except Exception as e:
            self.support_msg.set(f"Failed to start Mask Editor: {e}")
            self.mask_editor_button.state(['!disabled'])
            

    def check_process_status(self):
        if self.view_process:
            if self.view_process.poll() is not None:  # process ended
                self.view_button.state(['!disabled']) # Re-enable
                self.view_process = None
        if self.mask_edit_process:
            if self.mask_edit_process.poll() is not None:  # process ended
                self.mask_editor_button.state(['!disabled']) # Re-enable
                self.mask_edit_process = None
        self.left_frame.after(500, self.check_process_status)
        

    def apply_parameters(self):
        params_message = self.params_proto_snd.get_message_type()
        params_message.target_brightness = self.get_parameters(self.target_brightness_var, 'target_brightness', DEFAULT_TARGET_BRIGHTNESS)
        params_message.target_gain = self.get_parameters(self.target_gain_var, 'target_gain', DEFAULT_TARGET_GAIN)
        params_message.exposure_min = self.get_parameters(self.exposure_min_var, 'exposure_min', DEFAULT_EXPOSURE_MIN)
        params_message.exposure_max = self.get_parameters(self.exposure_max_var, 'exposure_max', DEFAULT_EXPOSURE_MAX)
        params_message.gain_min = self.get_parameters(self.gain_min_var, 'gain_min', DEFAULT_GAIN_MIN)
        params_message.gain_max = self.get_parameters(self.gain_max_var, 'gain_max', DEFAULT_GAIN_MAX)
        params_message.exposure_min_step = self.get_parameters(self.exposure_min_step_var, 'exposure_min_step', DEFAULT_EXPOSURE_MIN_STEP)
        params_message.gain_min_step = self.get_parameters(self.gain_min_step_var, 'gain_min_step', DEFAULT_GAIN_MIN_STEP)
        params_message.compensation_factor = self.get_parameters(self.compensation_factor_var, 'compensation_factor', DEFAULT_COMPENSATION_FACTOR)
        params_message.target_temperature = self.get_parameters(self.target_temperature_var, 'target_temperature', DEFAULT_TARGET_TEMPERATURE)
        params_message.histogram_sampling = self.get_parameters(self.histogram_sampling_var, 'histogram_sampling', DEFAULT_HISTOGRAM_SAMPLING)
        params_message.histogram_dark_point = self.get_parameters(self.histogram_dark_point_var, 'histogram_dark_point', DEFAULT_HISTOGRAM_DARK_POINT)
        params_message.histogram_bright_point = self.get_parameters(self.histogram_bright_point_var, 'histogram_bright_point', DEFAULT_HISTOGRAM_BRIGHT_POINT)
        
        params_message.reset_qhy = False
        params_message.close_qhy = False
        self.params_proto_snd.send(params_message)


    def shutdown(self):
        self.running = False


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
