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
        self.measurement = Measurement(file_path)
        self.queue = multiprocessing.Queue()
        self.broadcaster_list = []

    @staticmethod
    def _broadcaster(q, time_difference, channel, file_path):
        print(f"{channel} is broadcasting ...")

        measurement = Measurement(file_path)
        channel_content = measurement[channel]

        # Extract protobuf type from first message
        _, first_message = next(iter(channel_content))
        proto_type = type(first_message)

        # Initialize eCAL
        ecal_core.initialize(sys.argv, channel)
        pub = ProtoPublisher(channel, proto_type)

        for time_stamp, message in channel_content:
            curr_time = int(time.time() * 1.0e6) - time_difference
            wait_time = max(0, (time_stamp - curr_time) / 1.0e6)
            time.sleep(wait_time)
            pub.send(message)

        ecal_core.finalize()
        q.put(channel)

    def get_time_difference(self):
        rec_starting_time = min(
            next(iter(self.measurement[channel]))[0]
            for channel in self.measurement.channel_names
        )
        start_time = int(time.time() * 1.0e6) + 1.0e6
        return start_time - rec_starting_time

    def start(self):
        time_diff = self.get_time_difference()

        for channel in self.measurement.channel_names:
            proc = multiprocessing.Process(
                target=self._broadcaster,
                args=(self.queue, time_diff, channel, self.file_path)
            )
            self.broadcaster_list.append(proc)

        for proc in self.broadcaster_list:
            proc.start()

        for proc in self.broadcaster_list:
            proc.join()

        while not self.queue.empty():
            print(f"{self.queue.get()} ended")