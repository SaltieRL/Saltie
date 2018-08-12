from examples.autoencoder.autoencoder_model import AutoencoderModel
from examples.autoencoder.autoencoder_model_holder import AutoencoderModelHolder
from examples.autoencoder.autoencoder_output_formatter import AutoencoderOutputFormatter
from examples.legacy.legacy_input_formatter import LegacyInputFormatter
from examples.legacy.legacy_normalizer_input_formatter import LegacyNormalizerInputFormatter
from framework.output_formatter.host_output_formatter import HostOutputFormatter
from trainer.parsed_download_trainer import ParsedDownloadTrainer
import matplotlib.pyplot as plt


class PassLengthTrainer(ParsedDownloadTrainer):
    hit_lengths = []

    def process_file(self, input_file):
        print('Loading file ', input_file)
        steer = input_file.players[0].data.loc[1:100]['steer']
        if (steer > 0 & steer < 100).sum() > 0:
            print ('controller')

        ball_df = input_file.ball
        player_dfs = [p.data for p in input_file.players]
        hits = input_file.hits
        passes = list(filter(lambda k: k[1].analytics['pass_'], hits.items()))
        print(passes)
        for p in passes:
            p = p[1]
            length = p.next_hit.frame_number - p.frame_number
            self.hit_lengths.append(length)
        # train on hits

    def finish(self):
        plt.hist(self.hit_lengths, bins=50)
        plt.show()


if __name__ == '__main__':
    input_formatter = LegacyNormalizerInputFormatter(LegacyInputFormatter())
    output_formatter = HostOutputFormatter(AutoencoderOutputFormatter(input_formatter))
    pl = PassLengthTrainer(AutoencoderModelHolder(AutoencoderModel(compressed_dim=50),
                                                  input_formatter, output_formatter))
    # pl.train_on_files(500)
    pl.train_on_file(name='3E38E50B44101E81F91C40ABC99CA0AB')
