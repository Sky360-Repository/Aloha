# \copyright    Sky360.org
#
# \brief        Function to set a process name.
#               This is useful to identify different nodes running.
#
# ************************************************************************

import sys
import ecal.core.core as ecal_core


def set_process_name(process_name):
    ecal_core.initialize(sys.argv, process_name)
    ecal_core.set_process_state(1, 1, "ECAL running OK")
