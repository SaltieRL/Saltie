import argparse
import glob
import gzip
import os

import numpy as np

from bot_code.conversions import binary_converter

parser = argparse.ArgumentParser(
    description='Converts files in training/replays/ into files in training/converted/')
parser.add_argument('path', type=str,
                    help='replay directory to convert files from')
args = parser.parse_args()


class ReplayConverter:
    step = 0

    def __init__(self, game_file):
        self.game_file = game_file
        self.inputs = np.array([])
        self.outputs = np.array([])

    def process_pair(self, input_array, output_array, pair_number, file_version):
        self.inputs = np.append(self.inputs, np.array(input_array))
        self.outputs = np.append(self.outputs, np.array(output_array))
        if self.inputs.shape[0] % 2000 == 0:
            binary_converter.write_array_to_file(self.game_file, self.inputs)
            binary_converter.write_array_to_file(self.game_file, self.outputs)
            self.inputs = np.array([])
            self.outputs = np.array([])
            self.step = 0
        self.step += 1


fs = glob.glob(os.path.join(args.path, '*.gz'))
if not os.path.isdir('converted/'):
    os.makedirs('converted/')
print (fs, args.path)
for file in fs:
    with gzip.open(file, 'rb') as f:
        with gzip.open(os.path.join('converted', os.path.basename(file)), 'wb') as new_file:
            trainer = ReplayConverter(new_file)
            binary_converter.read_data(f, trainer.process_pair)
