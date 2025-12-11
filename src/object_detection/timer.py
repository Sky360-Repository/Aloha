# \copyright    Sky360.org
#
# \brief        Simple timer.
#
# ************************************************************************

import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name
        if name:
            print('[%s]' % name,)

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('Elapsed: %s' % (time.time() - self.tstart))
