import configparser
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
import importlib


MODEL_CONFIGURATION_HEADER = 'Model Configuration'
TRAINER_CONFIGURATION_HEADER = 'Trainer Configuration'


def download_batch(n):
    server = config.UPLOAD_SERVER
    r = requests.get(server + '/replays/list')
    replays = r.json()
    print('num replays available', len(replays), ' num requested ', n)
    n = min(n, len(replays))
    downloads = random.sample(replays, n)
    files = []
    download_counter = 0
    total_downloads = len(downloads)
    for f in downloads:
        r = requests.get(server + '/replays/' + f)
        files.append(io.BytesIO(r.content))
        if download_counter % 10 == 0:
            print('downloaded 10 more files: ', (float(download_counter) / float(total_downloads)) * 100.0)
        download_counter += 1
    print('downloaded all files')
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


def train_on_file(trainer_class, f):
    trainer_class.start_new_file()
    try:
        binary_converter.read_data(f, trainer_class.process_pair)
    except:
        print('error traiing on file')
    trainer_class.end_file()


def get_class(class_package, class_name):
    class_package = importlib.import_module(class_package)
    module_classes = inspect.getmembers(class_package, inspect.isclass)
    for class_group in module_classes:
        if class_group[0] == class_name:
            return class_group[1]
    return None


def load_config_file(config_file):
    if config_file is None:
        return
    #read file code here

    model_package = config_file.get(MODEL_CONFIGURATION_HEADER,
                                         'model_package')
    model_name = config_file.get(MODEL_CONFIGURATION_HEADER,
                                      'model_name')
    model_class = get_class(model_package, model_name)
    trainer_package = config_file.get(TRAINER_CONFIGURATION_HEADER,
                                    'trainer_package')
    trainer_name = config_file.get(TRAINER_CONFIGURATION_HEADER,
                                 'trainer_name')
    try:
        download_files = config_file.getboolean(TRAINER_CONFIGURATION_HEADER,
                                       'download_files')
    except Exception as e:
        download_files = True


    trainer_class = get_class(trainer_package, trainer_name)
    print('getting model from', model_package)
    print('name of model', model_name)

    if model_class is not None and trainer_class is not None:
        trainer_class.model_class = model_class
        model_class.config_file = config_file

    return trainer_class, download_files


if __name__ == '__main__':
    framework_config = configparser.RawConfigParser()
    framework_config.read('trainer.cfg')
    loaded_class, download_files = load_config_file(framework_config)
    trainer_object = loaded_class()
    files = get_all_files(download=download_files, n=2000)
    print('training on ' + str(len(files)) + ' files')
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
                        train_on_file(trainer_class=trainer_object, f=f)
                else:
                    # file on disk
                    with gzip.open(file, 'rb') as f:
                        train_on_file(trainer_class=trainer_object, f=f)
                end = time.time()
                difference = end - start
                total_time += difference
                print('trained file in ' + str(difference) + 's')
            except FileNotFoundError as e:
                print('whoops file not found')
                print(e.filename)
                print(file)
            except:
                print('error training on file going to next one')
    finally:
        print('ran through all files in ' + str(total_time / 60) + 'm')
        print('average time: ' + str((total_time / len(files))) + 's')
        trainer_object.end_everything()
