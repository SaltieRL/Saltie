import gzip
import io
import time

from bot_code.conversions import binary_converter
from bot_code.conversions.server_converter import ServerConverter
from bot_code.trainer.base_classes.base_trainer import BaseTrainer
from bot_code.trainer.utils.file_download_manager import get_file_get_function, get_file_list_get_function
from bot_code.trainer.utils.threaded_file_downloader import ThreadedFileDownloader


class DownloadTrainer(BaseTrainer):
    DOWNLOAD_TRAINER_CONFIGURATION_HEADER = 'Download Configuration'
    download_files = None
    max_files = None
    input_server = None
    get_file_function = None
    get_file_list_get_function = None
    download_manager = None
    num_downloader_threads = None
    num_trainer_threads = None
    should_batch_process = None

    def load_config(self):
        super().load_config()
        config = self.create_config()
        try:
            self.download_files = config.getboolean(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER,
                                                    'download_files')
        except Exception as e:
            self.download_files = True
        try:
            self.max_files = config.getint(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER, 'max_files')
        except Exception as e:
            self.max_files = 10000
        try:
            self.num_downloader_threads = config.getint(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER,
                                                        'number_download_threads')
        except Exception as e:
            self.num_downloader_threads = 10
        try:
            self.num_trainer_threads = config.getint(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER,
                                                     'number_training_threads')
        except Exception as e:
            self.num_trainer_threads = 1
        try:
            self.should_batch_process = config.getboolean(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER,
                                                   'batch_process')
        except Exception as e:
            self.should_batch_process = False

    def load_server(self):
        import config
        self.input_server = ServerConverter(config.UPLOAD_SERVER, False, False, False)

    def setup_trainer(self):
        """
        Sets up the downloader
        """
        super().setup_trainer()
        if self.download_files:
            self.load_server()
        self.get_file_function = get_file_get_function(self.download_files, self.input_server)
        self.get_file_list_get_function = get_file_list_get_function(self.download_files, self.input_server)
        self.download_manager = ThreadedFileDownloader(self.max_files, self.num_downloader_threads,
                                                       self.num_trainer_threads, self.get_file_list_get_function,
                                                       self.get_file_function, self.process_file)

    def _run_trainer(self):
        self.download_manager.create_and_run_workers()
        self.end_everything()

    def start_new_file(self):
        """
        Called when it is time to start training on a new file
        """
        pass

    def process_pair(self, input_array, output_array, pair_number, file_version):
        """
        Processes the pair of a single input, output array.
        :param input_array: This represents a single tick in the game
        :param output_array: This is the controls that the bot used during that same tick
        :param pair_number: Which tick number this is
        :param file_version: File version info
        :return:
        """
        pass

    def process_pair_batch(self, input_array, output_array, pair_number, file_version):
        """
        Processes a batch of input_array and output_array pairs
        :param input_array: This is a list of game ticks
        :param output_array: This is a list of controls that occurred.  Same length as input_array
        :param pair_number: Which tick number this batch started at
        :param file_version: File version info
        :return:
        """
        pass

    def end_file(self):
        """Called after all training on this file has completed"""
        pass

    def end_everything(self):
        """Called after all files have been trained and training is complete"""

    def train_file(self, file):
        self.start_new_file()
        if self.should_batch_process:
            try:
                binary_converter.read_data(file, self.process_pair_batch, batching=True)
            except Exception as e:
                print('error batch training on file ', e)
        else:
            try:
                binary_converter.read_data(file, self.process_pair, batching=False)
            except Exception as e:
                print('error training on file ', e)
        self.end_file()

    def process_file(self, input_file):
        """
        Opens a file and calls a function to train on the file.
        :param input_file: The file that has been loaded/downloaded
        :return: How long it took too process the file
        """
        if isinstance(input_file, io.BytesIO):
            input_file.seek(0)
        start = time.time()

        try:
            if isinstance(input_file, io.BytesIO):
                # file in memory
                with gzip.GzipFile(fileobj=input_file, mode='rb') as f:
                    self.train_file(f)
            else:
                # file on disk
                with gzip.open(input_file, 'rb') as f:
                    self.train_file(f)
        except FileNotFoundError as e:
            print('whoops file not found')
            print(e.filename)
            print(input_file)
        except Exception as e:
            print('error training on file going to next one ', e)

        end = time.time()
        difference = end - start
        print('trained file in', str(difference), '\bs\n\n')
        return difference
