from typing import List

from examples.autoencoder.autoencoder_model import AutoencoderModel
from examples.autoencoder.autoencoder_model_holder import AutoencoderModelHolder
from examples.autoencoder.autoencoder_output_formatter import AutoencoderOutputFormatter
from examples.legacy.legacy_input_formatter import LegacyInputFormatter
from examples.legacy.legacy_normalizer_input_formatter import LegacyNormalizerInputFormatter
from framework.output_formatter.host_output_formatter import HostOutputFormatter
from carball.analysis.saltie_game.saltie_game import Game
from trainer.parsed_download_trainer import ParsedDownloadTrainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PassLengthTrainer(ParsedDownloadTrainer):
    hit_lengths = []

    def process_file(self, input_file: Game):
        print('Loading file ', input_file)

        # Check to see if mouse + kb or controller
        steer = input_file.players[0].data.loc[1:100]['steer']
        if ((steer > 0) & (steer < 100)).sum() > 0:
            print('using controller')
        else:
            print('using keyboard')

        # Shot training
        ball_df = input_file.ball  # type: pd.DataFrame
        player_dfs = [p.data for p in input_file.players]  # type: List[pd.DataFrame]
        hits = input_file.hits
        # filter for shots (or whatever else you want [shot, pass_, dribble], from .analytics)
        shots = list(filter(lambda k: k[1].analytics['shot'], hits.items()))
        frames_back = 60  # how many frames to go back
        for frame_number, s in shots:
            ball_data = ball_df.loc[frame_number - frames_back:frame_number]
            # axis=1 concats all the columns, so you have one large vector of all the player information
            # (probably want to cut it down a bit, you don't need all the variables)
            player_data = pd.concat([p.loc[frame_number - frames_back:frame_number] for p in player_dfs],
                                    axis=1)
            # you can access .data here just like above, but this is the player who took the sho
            player_who_took_shot = s.player
            pass  # append somewhere?
        # train on hits

    def plot_grid(self, input_file, bars=49):
        bars = 49
        x_val = np.floor(((input_file.players[0].data.pos_x + 4120) / (8240 / bars)))
        y_val = np.floor(((input_file.players[0].data.pos_y + 5140) / ((5140 * 2) / bars)))
        pos_ = pd.concat([x_val.T, y_val.T], axis=1)
        pos_.columns = ['x', 'y']
        pos_ = pos_.dropna()
        plt.hist2d(pos_['x'], pos_['y'], bins=bars)
        plt.colorbar()
        plt.show()

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
