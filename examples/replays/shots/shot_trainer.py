from examples.replays.shots.shot_input_formatter import ShotInputFormatter
from examples.replays.shots.shot_output_formatter import ShotOutputFormatter
from framework.data_generator.local_cache_creator import LocalCacheCreator
from framework.data_generator.replay.hit_generator import HitGenerator

if __name__ == '__main__':
    hit_generator = HitGenerator()
    hit_generator.initialize(hit_filter={'shot': True})
    cache = LocalCacheCreator(ShotInputFormatter(), ShotOutputFormatter(), hit_generator)

    cache.create_cache()

    cache.save_cache('cache.ch')
