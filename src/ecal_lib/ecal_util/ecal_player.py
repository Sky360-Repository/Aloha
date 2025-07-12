# \copyright    Sky360.org
#
# \brief        Reads from hdf5 and broadcasts the messages.
#
# ************************************************************************

import sys
import time
import multiprocessing

from ecal.measurement.measurement import Measurement
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher


class EcalPlayer:
    def __init__(self, file_path: str):

        self.file_path = file_path

        # Create a measurement (pass either a .hdf5 file or a measurement folder)
        self.measurement = Measurement(self.file_path)
        self.queue = multiprocessing.Queue()
        self.broadcaster_list = []

    @staticmethod
    def _broadcaster(q, time_difference, channel, file_path):
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

    def get_time_difference(self):
        # Get starting time for this recoding
        rec_starting_time = sys.maxsize
        for channel in self.measurement.channel_names:
            channel_content = self.measurement[channel]
            # Get the first time stamp of each channel
            first_time_stamp = next(iter(channel_content))[0]
            rec_starting_time = min(rec_starting_time, first_time_stamp)

        # Giving 1ms to give time to set the processes
        start_time = int(time.time() * 1.0e6) + 1.0e6
        return start_time - rec_starting_time

    def start(self):
        time_difference = self.get_time_difference()
        for channel in self.measurement.channel_names:
            broadcaster_process = multiprocessing.Process(target=self._broadcaster,
                                                          args=(self.queue, time_difference, channel, self.file_path))
            self.broadcaster_list.append(broadcaster_process)

        for broadcaster_process in self.broadcaster_list:
            broadcaster_process.start()

        for broadcaster_process in self.broadcaster_list:
            broadcaster_process.join()

        while not self.queue.empty():
            print(f"{self.queue.get()} ended")
