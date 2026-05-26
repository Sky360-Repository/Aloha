# \copyright    Sky360.org
#
# \brief        Reads from hdf5 and broadcasts the messages.
#
# ************************************************************************

from src.ecal_player import EcalPlayer
from utils.hdf5_finder import HDF5Finder

if __name__ == "__main__":

    finder = HDF5Finder("/emmc/ecal_meas")
    hdf5_path = finder.ask_hdf5_files()
    player = EcalPlayer(hdf5_path)
    player.start()
