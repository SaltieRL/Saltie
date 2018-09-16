from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Text3D

from examples.autoencoder.autoencoder_model import AutoencoderModel
from examples.autoencoder.autoencoder_model_holder import AutoencoderModelHolder
from examples.autoencoder.autoencoder_output_formatter import AutoencoderOutputFormatter
from examples.legacy.legacy_input_formatter import LegacyInputFormatter
from examples.legacy.legacy_normalizer_input_formatter import LegacyNormalizerInputFormatter
from framework.output_formatter.host_output_formatter import HostOutputFormatter
from carball.analysis.saltie_game.metadata.ApiPlayer import Player
from trainer.parsed_download_trainer import ParsedDownloadTrainer
import matplotlib.pyplot as plt

import pandas as pd


class Visualizer(ParsedDownloadTrainer):
    hit_lengths = []

    def process_file(self, input_file):
        num_players = len(input_file.players)
        print('Loading file ', input_file)
        ball_df = input_file.ball
        player_dfs = [p.data for p in input_file.players]
        hits = input_file.hits
        passes = list(filter(lambda k: k[1].analytics['pass_'], hits.items()))
        for p in passes:
            p = p[1]
            length = p.next_hit.frame_number - p.frame_number
            self.hit_lengths.append(length)
        fig = plt.figure()
        # ax = p3.Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')

        # set limits
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_zlim(0, 2000)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # draw field lines
        ax.plot([-5000, 5000], [-5000, -5000])
        ax.plot([-5000, 5000], [5000, 5000])

        # set colors
        colors = []
        for pl in input_file.players:  # type: Player
            if pl.is_orange:
                colors.append('#FFA500')
            else:
                colors.append('#0000FF')
        colors.append('#00FF00') # green for ball

        # create initial points
        graphs = []
        num_pts = num_players + 1
        markers = ["x"] * num_players + ["o"]
        for n in range(num_pts):
            graph, = ax.plot([0.], [0.], [0.], linestyle="",
                             marker=markers[n], color=colors[n], ms=10)
            graphs.append(graph)

        # create player labels
        players = []
        for n in range(num_players):
            txt = ax.text(0., 0., 0., input_file.players[n].name, 'z')
            players.append(txt)

        def update(num):
            for i, p in enumerate(player_dfs):  # type: pd.DataFrame
                if num not in p.index.values:
                    continue
                x = p['pos_x'].loc[num]
                y = p['pos_y'].loc[num]
                z = p['pos_z'].loc[num]
                graphs[i].set_data(x, y)
                graphs[i].set_3d_properties(z)
                txt = players[i]  # type: Text3D
                txt.set_position((x, y))
                txt.set_3d_properties(z + 100)
                players[i] = txt
            if num in ball_df.index.values:
                x = ball_df['pos_x'].loc[num]
                y = ball_df['pos_y'].loc[num]
                z = ball_df['pos_z'].loc[num]
                graphs[-1].set_data(x, y)
                graphs[-1].set_3d_properties(z)
            return None, graphs,
        frames = len(player_dfs[0])
        anim = animation.FuncAnimation(fig, update, frames, interval=30, blit=False)
        # anim.save('line.gif', dpi=80, writer='imagemagick')
        plt.show()
        # train on hits

    def finish(self):
        plt.hist(self.hit_lengths, bins=50)
        plt.show()


if __name__ == '__main__':
    input_formatter = LegacyNormalizerInputFormatter(LegacyInputFormatter())
    output_formatter = HostOutputFormatter(AutoencoderOutputFormatter(input_formatter))
    pl = Visualizer(AutoencoderModelHolder(AutoencoderModel(compressed_dim=50),
                                           input_formatter, output_formatter))
    pl.train_on_file()
