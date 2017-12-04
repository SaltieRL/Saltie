import gzip
import inspect
import io
import os
import random
import sys
import time

import requests

import config

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from conversions import binary_converter
from trainer.model_eval_trainer import EvalTrainer


def download_batch(n):
    server = config.UPLOAD_SERVER
    r = requests.get(server + '/replays/list')
    replays = r.json()
    print('num replays available', len(replays), ' num requested ', n)
    n = min(n, len(replays))
    downloads = random.sample(replays, n)
    files = []
    for f in downloads:
        r = requests.get(server + '/replays/' + f)
        files.append(io.BytesIO(r.content))
    return files


def get_all_files(download=False, n=5):
    if download:
        return download_batch(n)
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    training_path = dir_path + '\\training'
    files = []
    include_extensions = {'gz'}
    exclude_paths = {'data', 'ignore'}
    exclude_files = {''}
    for (dirpath, dirnames, filenames) in os.walk(training_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_paths]
        for file in filenames:
            skip_file = False
            if file.split('.')[-1] not in include_extensions:
                continue
            for excluded_name in exclude_files:
                if excluded_name in file and excluded_name != '':
                    print('exclude file: ' + file)
                    skip_file = True
                    break
            if skip_file:
                continue
            files.append(dirpath + '\\' + file)
    return files


def get_trainer_class():
    # fill your input function here!
    return EvalTrainer


def train_on_file(trainerClass, f):
    trainerClass.start_new_file()
    binary_converter.read_data(f, trainerClass.process_pair)
    trainerClass.end_file()


if __name__ == '__main__':
    files = get_all_files(download=True)
    print('training on ' + str(len(files)) + ' files')
    trainerClass = get_trainer_class()()
    counter = 0
    total_time = 0
    try:
        for file in files:
            if isinstance(file, io.BytesIO):
                file.seek(0)
            start = time.time()
            counter += 1
            try:
                print('running file ' + str(counter) + '/' + str(len(files)))
                if isinstance(file, io.BytesIO):
                    # file in memory
                    with gzip.GzipFile(fileobj=file, mode='rb') as f:
                        train_on_file(trainerClass=trainerClass, f=f)
                else:
                    # file on disk
                    with gzip.open(file, 'rb') as f:
                        train_on_file(trainerClass=trainerClass, f=f)
                end = time.time()
                difference = end - start
                total_time += difference
                print('trained file in ' + str(difference) + 's')
            except FileNotFoundError as e:
                print('whoops file not found')
                print(e.filename)
                print(file)
    finally:
        print('ran through all files in ' + str(total_time / 60) + 'm')
        print('average time: ' + str((total_time / len(files))) + 's')
        trainerClass.end_everything()
