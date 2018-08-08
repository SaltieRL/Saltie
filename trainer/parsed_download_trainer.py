from examples.autoencoder.autoencoder_model import AutoencoderModel
from examples.autoencoder.autoencoder_model_holder import AutoencoderModelHolder
from examples.autoencoder.autoencoder_output_formatter import AutoencoderOutputFormatter
from examples.legacy.legacy_input_formatter import LegacyInputFormatter
from examples.legacy.legacy_normalizer_input_formatter import LegacyNormalizerInputFormatter
from framework.model_holder.base_model_holder import BaseModelHolder
from framework.output_formatter.host_output_formatter import HostOutputFormatter
from trainer.base_trainer import BaseTrainer
from trainer.downloader import Downloader
import matplotlib.pyplot as plt


class ParsedDownloadTrainer(BaseTrainer):
    def __init__(self, model_holder: BaseModelHolder):
        super().__init__(model_holder)
        self.downloader = Downloader()

    def process_file(self, input_file):
        pass

    def train_on_file(self, name=None):
        if name is None:
            input_file = self.downloader.download_pandas_game(from_disk=False)
        else:
            input_file = self.downloader.download_pandas_game(hash=name)
        self.process_file(input_file)

    def train_on_files(self, count=200):
        counter = 0
        for i in range(count):
            input_file = self.downloader.download_pandas_game(from_disk=False)
            self.process_file(input_file)
            counter += 1
            if counter % 10 == 0:
                print('FILE', counter)

    def finish(self):
        self.model_holder.finish_training()


if __name__ == '__main__':
    input_formatter = LegacyNormalizerInputFormatter(LegacyInputFormatter())
    output_formatter = HostOutputFormatter(AutoencoderOutputFormatter(input_formatter))
    d = ParsedDownloadTrainer(AutoencoderModelHolder(AutoencoderModel(compressed_dim=50),
                                                     input_formatter, output_formatter))
    # d.train_on_files()
    d.train_on_file()
    d.finish()
