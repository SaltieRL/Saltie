import io
import json
import random
import zipfile

import requests

from model_holder.base_model_holder import BaseModelHolder
from trainer.base_trainer import BaseTrainer


class DownloadTrainer(BaseTrainer):
    BASE_URL = "http://saltie.tk:5000"

    def __init__(self, model_holder: BaseModelHolder):
        super().__init__(model_holder)

    @staticmethod
    def unzip(in_memory_file: io.BytesIO):
        in_memory_zip_file = zipfile.ZipFile(in_memory_file)
        return [io.BytesIO(in_memory_zip_file.read(name)) for name in in_memory_zip_file.namelist()]

    @staticmethod
    def create_in_memory_file(response: requests.Response) -> io.BytesIO:
        in_memory_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                in_memory_file.write(chunk)
        return in_memory_file

    def get_random_replay(self):
        js = requests.get(self.BASE_URL + '/replays/list').json()
        filename = random.choice(js)
        return self.get_replay(filename)

    def get_replay(self, filename_or_filenames: list or str):
        if isinstance(filename_or_filenames, list):
            r = requests.post(self.BASE_URL + '/replays/download', data={'files': json.dumps(filename_or_filenames)})
            imf = self.create_in_memory_file(r)
            return self.unzip(imf)
        else:
            r = requests.get(self.BASE_URL + f'/replays/{filename_or_filenames}')
            imf = self.create_in_memory_file(r)
            return imf