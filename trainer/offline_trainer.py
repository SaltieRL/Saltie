import configparser
import gzip
import inspect
import io
import os

import sys
import time

from conversions.server_converter import ServerConverter
import config


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from conversions import binary_converter
from trainer.threaded_trainer import ThreadedFiles
import importlib


MODEL_CONFIGURATION_HEADER = 'Model Configuration'
TRAINER_CONFIGURATION_HEADER = 'Trainer Configuration'


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


def get_class(class_package, class_name):
    class_package = importlib.import_module(class_package)
    module_classes = inspect.getmembers(class_package, inspect.isclass)
    for class_group in module_classes:
        if class_group[0] == class_name:
            return class_group[1]
    return None


def get_all_files(max_files, only_eval):
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


def train_file(trainer_class, f):
    trainer_class.start_new_file()
    try:
        binary_converter.read_data(f, trainer_class.process_pair)
    except Exception as e:
        print('error training on file ', e)
    trainer_class.end_file()


def train_with_file(input_file, train_object):
    if isinstance(input_file, io.BytesIO):
        input_file.seek(0)
    start = time.time()

    try:

        if isinstance(input_file, io.BytesIO):
            # file in memory
            with gzip.GzipFile(fileobj=input_file, mode='rb') as f:
                train_file(trainer_class=train_object, f=f)
        else:
            # file on disk
            with gzip.open(input_file, 'rb') as f:
                train_file(trainer_class=train_object, f=f)
    except FileNotFoundError as e:
        print('whoops file not found')
        print(e.filename)
        print(input_file)
    except Exception as e:
        print('error training on file going to next one ', e)

    end = time.time()
    difference = end - start
    print('trained file in ' + str(difference) + 's')
    return difference


def get_file_get_function(download, input_server):
    if download:
        return input_server.download_file
    else:
        return lambda input_file: input_file


def get_file_list_get_function(download, input_server):
    if download:
        return input_server.get_replays
    else:
        return get_all_files



if __name__ == '__main__':
    framework_config = configparser.RawConfigParser()
    framework_config.read('trainer.cfg')
    loaded_class, should_download_files = load_config_file(framework_config)
    trainer_instance = loaded_class()

    server = ServerConverter(config.UPLOAD_SERVER, False, False, False)

    max_files = 100
    num_download_threads = 5
    num_train_threads = 1

    file_threader = ThreadedFiles(max_files,
                                  num_download_threads,
                                  num_train_threads,
                                  get_file_list_get_function(should_download_files, server),
                                  get_file_get_function(should_download_files, server),
                                  train_with_file,
                                  trainer_instance)

    file_threader.create_and_run_workers()
