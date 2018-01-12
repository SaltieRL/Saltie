import configparser
import gzip

import io
import time

from conversions.server_converter import ServerConverter
import config
from conversions import binary_converter
from trainer.utils.threaded_file_downloader import ThreadedFileDownloader




if __name__ == '__main__':
    framework_config = configparser.RawConfigParser()
    framework_config.read('trainer.cfg')
    loaded_class, should_download_files = load_config_file(framework_config)
    trainer_instance = loaded_class()

    server = ServerConverter(config.UPLOAD_SERVER, False, False, False)

    max_files = 10000
    num_download_threads = 10
    num_train_threads = 1

    file_threader = ThreadedFileDownloader(max_files,
                                           num_download_threads,
                                           num_train_threads,
                                           get_file_list_get_function(should_download_files, server),
                                           get_file_get_function(should_download_files, server),
                                           train_with_file,
                                           trainer_instance)

    file_threader.create_and_run_workers()
