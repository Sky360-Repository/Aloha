# \copyright    Sky360.org
#
# \brief        Reads from hdf5 and broadcasts the messages.
#
# ************************************************************************

import os
import sys
import time
import glob
import multiprocessing
import tkinter as tk
from tkinter import filedialog

from ecal.measurement.measurement import Measurement
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher


def path_as_hdf5_files(directory_path):
    # Use glob to find all .hdf5 files in the directory_path
    return glob.glob(directory_path + '/*.hdf5')


def find_hdf5_files(file_path):
    # If any .hdf5 files are found, return the path to the directory_path
    if path_as_hdf5_files(file_path):
        return file_path

    # Iterate over all subdirectories of file_path
    for root, dirs, files in os.walk(file_path):
        for sub_dir in dirs:
            # Construct the path to the subdirectory
            subdirectory_path = os.path.join(root, sub_dir)
            # If any .hdf5 files are found, return the path to the directory_path
            if path_as_hdf5_files(subdirectory_path):
                return subdirectory_path

    # If no .hdf5 files are found in any subdirectory, return None
    return None


def ask_hdf5_files():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory(initialdir="C:/ecal_meas/", title="Select a folder with .hdf5 or .ecalmeas")
    # Look for hdf5 files
    file_path = find_hdf5_files(file_path)
    return file_path


def broadcaster(q, time_difference, channel, file_path):

    print(f"{channel} is broadcasting ...")

    measurement = Measurement(file_path)
    channel_content = measurement[channel]

    # measurement for the channel
    proto_pb = next(iter(channel_content))[1]

    # initialize eCAL API.
    ecal_core.initialize(sys.argv, channel)

    # Create a Publisher
    pub = ProtoPublisher(channel, proto_pb)

    for (time_stamp, message) in channel_content:
        # Current time adjusted to the recording
        curr_time = int(time.time() * 1.0e6) - time_difference
        wait_time = max(0, (time_stamp - curr_time) / 1.0e6)
        # Wait for the time to broadcast
        time.sleep(wait_time)

        # broadcast
        pub.send(message)

    # finalize eCAL API
    ecal_core.finalize()
    # Signal that the process terminated
    q.put(channel)


def ecal_player():
    file_path = ask_hdf5_files()
    if file_path is None:
        raise Exception("This folder doesn't seem to have valid ecal measurements")

    # Create a measurement (pass either a .hdf5 file or a measurement folder)
    measurement = Measurement(file_path)

    # Get starting time for this recoding
    rec_starting_time = sys.maxsize
    for channel in measurement.channel_names:
        channel_content = measurement[channel]
        # Get the first time stamp of each channel
        first_time_stamp = next(iter(channel_content))[0]
        rec_starting_time = min(rec_starting_time, first_time_stamp)

    # Giving 1ms to give time to set the processes
    start_time = int(time.time() * 1.0e6) + 1.0e6
    time_difference = start_time - rec_starting_time

    queue = multiprocessing.Queue()
    broadcaster_list = []
    for channel in measurement.channel_names:
        broadcaster_process = multiprocessing.Process(target=broadcaster,
                                                      args=(queue, time_difference, channel, file_path))
        broadcaster_list.append(broadcaster_process)

    for broadcaster_process in broadcaster_list:
        broadcaster_process.start()

    for broadcaster_process in broadcaster_list:
        broadcaster_process.join()

    while not queue.empty():
        print(f"{queue.get()} ended")


if __name__ == "__main__":
    ecal_player()
