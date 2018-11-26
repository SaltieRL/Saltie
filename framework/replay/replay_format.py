import zlib
from io import BytesIO

from carball.analysis.utils.pandas_manager import PandasManager
from carball.generated.api import game_pb2
import pandas as pd
from carball.generated.api.stats import events_pb2


class GeneratedReplay:
    def __init__(self, protobuf, pandas):
        """
        :param protobuf: described in https://github.com/SaltieRL/carball/tree/master/api
        :param pandas: contains the raw frame data of a replay
        """
        self.pandas = pandas
        self.protobuf = protobuf
        self.decoded_proto = None
        self.decoded_pandas = None

    def get_proto(self) -> game_pb2.Game:
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
        return self.decoded_proto

    def get_pandas(self) -> pd.DataFrame:
        """
        Gets the pandas lazily decompiles it
        :return:
        """
        if self.decoded_pandas is not None:
            return self.decoded_pandas
        stream = BytesIO(zlib.decompress(self.pandas, zlib.MAX_WBITS | 16))
        self.decoded_pandas = PandasManager.safe_read_pandas_to_memory(stream)
        self.pandas = None
        return self.decoded_pandas


class GeneratedHit:
    def __init__(self, hit: events_pb2.Hit, replay: GeneratedReplay):
        self.replay = replay
        self.hit = hit

    def get_hit(self) -> events_pb2.Hit:
        return self.hit

    def get_game_state(self):
        """
        :return: Game state at a given point in time
        """
        return None

    def get_replay(self) -> GeneratedReplay:
        return self.replay
