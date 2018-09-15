import zlib

from carball.analysis.utils.pandas_manager import PandasManager
from carball.generated.api import game_pb2


class Replay:
    def __init__(self, protobuf, pandas):
        """
        :param protobuf: described in https://github.com/SaltieRL/carball/tree/master/api
        :param pandas: contains the raw frame data of a replay
        """
        self.pandas = pandas
        self.protobuf = protobuf
        self.decoded_proto = None
        self.decoded_pandas = None

    def get_proto(self):
        """
        Gets the protobuf lazily decompiles it
        :return:
        """
        if self.decoded_proto is not None:
            return self.decoded_proto
        self.decoded_proto = game_pb2.Game()
        self.decoded_proto.ParseFromString(self.protobuf)
        # free up the memory
        self.protobuf = None

    def get_pandas(self):
        """
        Gets the pandas lazily decompiles it
        :return:
        """
        if self.decoded_pandas is not None:
            return self.decoded_pandas
        self.decoded_pandas = PandasManager.safe_read_pandas_to_memory(zlib.decompress(self.pandas))
        self.pandas = None
