import io
import json
import os

import pandas
import pickle
import random
import zipfile

import fs
import requests
import sys

from requests.exceptions import ChunkedEncodingError

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'framework', 'replayanalysis'))  # dirty way to fix the path for the submodule pickling


class Downloader:
    BASE_URL = 'http://138.197.6.71:5000'  # for saltie replays/training
    BASE_REPLAY_URL = "http://saltie.tk"  # for replay parsing/training
    API_KEY = '123456'

    def __init__(self, max_size_mb=100, path='mem://saltie'):
        self.max_size_mb = max_size_mb
        self.filesystem = fs.open_fs(path)

    @staticmethod
    def unzip(in_memory_file: io.BytesIO):
        in_memory_zip_file = zipfile.ZipFile(in_memory_file)
        return [io.BytesIO(in_memory_zip_file.read(name)) for name in in_memory_zip_file.namelist()]

    @staticmethod
    def create_in_memory_file(response: requests.Response) -> io.BytesIO:
        in_memory_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            print('chunk')
            if chunk:
                in_memory_file.write(chunk)
        return in_memory_file

    def get_random_replay(self):
        js = requests.get(self.BASE_URL + '/replays/list?model_hash=rashbot0').json()
        filename = random.choice(js)
        return self.get_replay(filename), filename

    def get_replays(self, number=1, batch=50):
        batch = min(number, batch)
        js = requests.get(self.BASE_URL + '/replays/list?model_hash=rashbot0').json()
        filenames = []
        file_list = []

        total_filenames = random.sample(js, number)
        for i in range(int(number / batch)):
            sequence_filenames = total_filenames[i * batch: (i + 1) * batch]
            file_list += self.get_replay(sequence_filenames)
            filenames += sequence_filenames
            print('downloaded', (batch * (i + 1.0)) / number * 100, '% of files')
        return zip(file_list, filenames)

    def get_replay(self, filename_or_filenames: list or str):
        if isinstance(filename_or_filenames, list):
            try:
                r = requests.post(self.BASE_URL + '/replays/download',
                                  data={'files': json.dumps(filename_or_filenames)})
            except ChunkedEncodingError:
                return []
            imf = self.create_in_memory_file(r)
            return self.unzip(imf)
        else:
            r = requests.get(self.BASE_URL + '/replays/{}'.format(filename_or_filenames))
            imf = self.create_in_memory_file(r)
            return imf

    def download_replays(self):
        rpl, fn = self.get_random_replay()
        success = self.filesystem.create(fn)
        if success:
            # file has been successfully created
            self.filesystem.setfile(fn, rpl)

    def download_pandas_game(self, from_disk=False, hash=None) -> pandas.DataFrame:
        if not from_disk:
            if hash is None:
                js = requests.get(self.BASE_REPLAY_URL + '/api/v1/parsed/list?key={}'.format(self.API_KEY)).json()
                dl = random.choice(js)
            else:
                dl = hash + '.replay.pkl'
            dl_url = self.BASE_REPLAY_URL + '/api/v1/parsed/{}?key={}'.format(dl, self.API_KEY)
            r = requests.get(dl_url, stream=True)
            r.raw.decode_content = True  # Content-Encoding
            r.raise_for_status()
            try:
                game = pickle.load(io.BytesIO(r.content))
            except (EOFError, ImportError):
                return self.download_pandas_game(from_disk=False)
        else:
            game = pickle.load(open('test.pkl', 'rb'))
        return game


if __name__ == '__main__':
    dl = Downloader()
    # dl.download_replays()
    # print(dl.filesystem.listdir('/'))
    game = dl.download_pandas_game(True)
    print()
