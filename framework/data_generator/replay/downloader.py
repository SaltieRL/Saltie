import requests

from framework.data_generator.base_generator import BaseDataGenerator
from framework.replay.replay import Replay


class ReplayListGenerator(BaseDataGenerator):

    BASE_URL = "https://calculated.gg"

    def __init__(self, api_key=1, min_mmr=0, max_mmr=4000, num_players_on_team=-1, max_pages=3, shuffle=False):
        super().__init__()
        self.api_key = api_key
        self.min_mmr = min_mmr
        self.max_mmr = max_mmr
        self.num_players_on_team = num_players_on_team
        self.max_pages = max_pages

        self.shuffle = shuffle
        self.next_page = True
        self.existing_url = '/api/v1/replays?page=1&key=' + str(api_key)
        self.replays = []

    def initialize(self, **kwargs):
        pass

    def create_url(self, existing_url):
        return self.BASE_URL + existing_url +'&minmmr=' + str(self.min_mmr) + '&max_mmr=' + str(self.max_mmr)

    def has_next(self):
        return len(self.replays) > 0 or self.next_page

    def _next(self):
        if len(self.replays) > 0:
            return self.replays.pop()
        self.logger.debug('getting list of replays')
        js = requests.get(self.create_url(self.existing_url)).json()
        next_url = js['next']
        if next_url is not None and next_url != '' and js['page'] < self.max_pages:
            self.logger.debug('loading next page: %s', next_url)
            self.existing_url = next_url
        else:
            self.logger.debug('now more pages to load')
            self.next_page = False
        replay_list = js['data']
        self.replays = [replay['hash'] for replay in replay_list
                        if replay['teamsize'] == self.num_players_on_team or self.num_players_on_team == -1]
        self.logger.info('%s replays in queue', len(self.replays))
        return self.replays.pop()


class CalculatedDownloader(ReplayListGenerator):

    DOWNLOAD_URL = "/api/v1/parsed/"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = []
        self.buffer_size = 10
        self.parallel_threads = 1
        self.key_url = None

    def initialize(self, buffer_size, parallel_threads, **kwargs):
        super().initialize(**kwargs)
        self.buffer_size = buffer_size
        self.parallel_threads = parallel_threads
        self.key_url = '?key=' + str(self.api_key)

    def has_next(self):
        return len(self.buffer) > 0 or super().has_next()

    def do_threads(self):
        pass

    def download_replay(self, replay_hash):
        # get pts
        pts = requests.get(self.BASE_URL + self.DOWNLOAD_URL + replay_hash + '.replay.pts' + self.key_url)
        pandas = requests.get(self.BASE_URL + self.DOWNLOAD_URL + replay_hash + '.replay.gzip' + self.key_url)
        replay = Replay(protobuf=pts.content, pandas=pandas.content)
        return replay

    def _next(self):
        if len(self.buffer) > 0:
            return self.buffer.pop()
        replay = self.download_replay(super()._next())
        return replay


if __name__ == "__main__":
    import logging
    # https://calculated.gg/api/v1/parsed/1097A28E46D0756EEB7820BFD31BE226.replay.pts?key=1
    downloader = CalculatedDownloader()
    downloader.logger.setLevel(logging.DEBUG)
    downloader.initialize(buffer_size=10, parallel_threads=1)
    count = 1
    for i in downloader.get_data():
        print(str(count))
        count += 1
