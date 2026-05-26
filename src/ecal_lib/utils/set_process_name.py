# \copyright    Sky360.org
#
# \brief        Function to set a process name.
#               This is useful to identify different nodes running.
#
# ************************************************************************

import sys
import ecal.core.core as ecal_core

_INITIALISED = False

def set_process_name(process_name: str):
    global _INITIALISED

    if not _INITIALISED:
        ecal_core.initialize(sys.argv, process_name)
        _INITIALISED = True

    # Always keep state healthy
    ecal_core.set_process_state(1, 1, "ECAL running OK")

