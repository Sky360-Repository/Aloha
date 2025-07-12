# \copyright    Sky360.org
#
# \brief        Finds folder with hdf5.
#
# ************************************************************************

import glob
import os
import tkinter as tk
from tkinter import filedialog


class HDF5Finder:
    def __init__(self, initial_dir="C:/ecal_meas/"):
        self.initial_dir = initial_dir

    @staticmethod
    def path_as_hdf5_files(directory_path):
        """Return a list of .hdf5 files in the given directory."""
        return glob.glob(directory_path + '/*.hdf5')

    def find_hdf5_files(self, file_path):
        """Find a directory containing .hdf5 files, searching recursively if needed."""
        if self.path_as_hdf5_files(file_path):
            return file_path

        for root, dirs, files in os.walk(file_path):
            for sub_dir in dirs:
                subdirectory_path = os.path.join(root, sub_dir)
                if self.path_as_hdf5_files(subdirectory_path):
                    return subdirectory_path

        return None  # No valid .hdf5 files found

    def ask_hdf5_files(self):
        """Prompt user to select a directory and ensure it contains .hdf5 files."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory(initialdir=self.initial_dir,
                                            title="Select a folder with .hdf5 or .ecalmeas")

        file_path = self.find_hdf5_files(file_path)
        if file_path is None:
            raise Exception("This folder doesn't seem to have valid eCAL measurements")

        return file_path
