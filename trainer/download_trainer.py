import gzip
import io

from input_formatter.base_input_formatter import BaseInputFormatter
from model.base_model import BaseModel
from model_holder.base_model_holder import BaseModelHolder
from trainer.base_trainer import BaseTrainer
from trainer.downloader import Downloader
import trainer.binary_converter as bc


class DownloadTrainer(BaseTrainer):
    def __init__(self, model_holder: BaseModelHolder):
        super().__init__(model_holder)
        self.downloader = Downloader()

    def train_on_file(self):
        input_file = self.downloader.get_random_replay()
        file_name = input_file[1]
        input_file = input_file[0]
        if isinstance(input_file, io.BytesIO):
            input_file.seek(0)
            with gzip.GzipFile(fileobj=input_file, mode='rb') as f:
                bc.read_data(f, self.model_holder.process_pair)


if __name__ == '__main__':
    d = DownloadTrainer(BaseModelHolder(BaseModel(), BaseInputFormatter()))
    d.train_on_file()
