import random
from typing import Dict

from framework.data_generator.replay.replay_generator import ReplayDownloaderGenerator
from framework.replay.replay_format import GeneratedHit


class HitGenerator(ReplayCacheGenerator):

    def __init__(self, hit_filter:Dict=None **kwargs):
        super().__init__(**kwargs)
        self.hit_filter = None
        self.hit_buffer = []
        self.current_replay = None
        self.hit_filter = hit_filter

    def has_next(self):
        return len(self.hit_buffer) > 0 or super().has_next()

    def filter_hit(self, hit):
        for key, filter in self.hit_filter.items():
            if getattr(hit, key) != filter:
                return False
        return True

    def __get_next_hit(self):
        while len(self.hit_buffer) > 0:
            if self.hit_filter is not None:
                next_hit = self.hit_buffer.pop()
                if self.filter_hit(next_hit):
                    return GeneratedHit(next_hit, self.current_replay)
            else:
                return GeneratedHit(self.hit_buffer.pop(), self.current_replay)
        self.current_replay = super()._next()
        proto = self.current_replay.get_proto()
        hits = proto.game_stats.hits
        self.hit_buffer = [proto_hit for proto_hit in hits]
        self.logger.info('%s hits in queue', len(self.hit_buffer))
        if self.shuffle and len(self.hit_buffer) > 0:
            random.shuffle(self.replays)
        return self.__get_next_hit()

    def _next(self) -> GeneratedHit:
        return self.__get_next_hit()


if __name__ == "__main__":
    # https://calculated.gg/api/v1/parsed/1097A28E46D0756EEB7820BFD31BE226.replay.pts?key=1
    hit_creator = HitGenerator(max_pages=1)
    hit_creator.initialize(buffer_size=10, parallel_threads=1, cache_size=100, hit_filter={'goal': True})
    count = 1
    for hit in hit_creator.get_data():
        print(str(count))
        count += 1
