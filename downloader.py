import trainer.offline_trainer as ot

import os

import configparser
from conversions.server_converter import ServerConverter
import config
from trainer.threaded_trainer import ThreadedFiles

config_file_name = os.path.join(os.getcwd(), 'trainer/trainer.cfg')
framework_config = configparser.RawConfigParser()
framework_config.read(config_file_name)
loaded_class, should_download_files = ot.load_config_file(framework_config)
trainer_instance = loaded_class()

server = ServerConverter(config.UPLOAD_SERVER, False, False, False)

max_files = 6000
num_download_threads = 2
num_train_threads = 1

file_threader = ThreadedFiles(max_files,
                              num_download_threads,
                              num_train_threads,
                              ot.get_file_list_get_function(should_download_files, server),
                              ot.get_file_get_function(should_download_files, server),
                              ot.train_with_file,
                              trainer_instance)

file_threader.create_and_run_workers()
