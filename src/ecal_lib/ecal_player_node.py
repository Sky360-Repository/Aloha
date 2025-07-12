# \copyright    Sky360.org
#
# \brief        Reads from hdf5 and broadcasts the messages.
#
# ************************************************************************

from ecal_util.ecal_player import EcalPlayer
from ecal_util.hdf5_finder import HDF5Finder

if __name__ == "__main__":

    finder = HDF5Finder("/emmc/ecal_meas")
    hdf5_path = finder.ask_hdf5_files()
    player = EcalPlayer(hdf5_path)
    player.start()
