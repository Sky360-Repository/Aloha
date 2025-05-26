#!/bin/bash

# Set the LD_PRELOAD variable
#export LD_PRELOAD=/opt/libusb-1.0.18/lib/libusb-1.0.so.0.1.0
#export LD_PRELOAD=/opt/libusb-1.0.19/lib/libusb-1.0.so.0.1.0
#export LD_PRELOAD=/opt/libusb-1.0.20/lib/libusb-1.0.so.0.1.0
#export LD_PRELOAD=/opt/libusb-1.0.21/lib/libusb-1.0.so.0.1.0
export LD_PRELOAD=/opt/libusb-1.0.22/lib/libusb-1.0.so.0.1.0
#export LD_PRELOAD=/opt/libusb-1.0.23/lib/libusb-1.0.so.0.2.0
#export LD_PRELOAD=/opt/libusb-1.0.24/lib/libusb-1.0.so.0.3.0
#export LD_PRELOAD=/opt/libusb-1.0.25/lib/libusb-1.0.so.0.3.0

# Start the Python script
python3 ../src/camera_controler/qhy_controller//QHYCameraController.py

