import hashlib
from models.base_model import BaseModel


MODEL_CONFIGURATION_HEADER = 'Model Configuration'


class DownloaderModel(BaseModel):
    output_vector_actions = ['throttle', 'steer', 'pitch', 'yaw',
                             'roll', 'jump', 'boost', 'handbrake']

    config_file = None
    is_initialized = False
    model_file = None
    is_evaluating = False
    is_online_training = False
    # no_op = tf.no_op()
    # train_op = no_op

    """"
    This is a base class for all models It has a couple helper methods but is mainly used to provide a standard
    interface for running and training a model
    """

    def __init__(self, is_training=False):
        self.all_inputs = []
        self.all_outputs = []

        self.is_training = is_training

        if self.config_file is not None:
            self.load_config_file()

    def initialize_model(self):
        self.is_initialized = True

    def get_model_name(self):
        """
        :return: The name of the model used for saving the file
        """
        return 'Downloader Model'

    # def get_default_file_name(self):
    #     return 'trained_variables'

    # def get_model_path(self, filename):
    #     """
    #     Creates a path for saving a file, this puts it in the directory of [get_model_name]
    #     :param filename: name of the file being saved
    #     :return: The path of the file
    #     """
    #     dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     return dir_path + "/training/data/" + self.get_model_name() + "/" + filename

    def load_config_file(self):
        try:
            self.model_file = self.config_file.get(
                MODEL_CONFIGURATION_HEADER, 'model_directory', fallback='downloader_model')
        except Exception as e:
            print('model directory is not in config', e)

        try:
            self.batch_size = self.config_file.getint(
                MODEL_CONFIGURATION_HEADER, 'batch_size', fallback=self.batch_size)
        except Exception:
            print('batch size is not in config')

        try:
            self.mini_batch_size = self.config_file.getint(
                MODEL_CONFIGURATION_HEADER, 'mini_batch_size', fallback=self.mini_batch_size)
        except Exception:
            print('mini batch size is not in config')

        try:
            self.is_evaluating = self.config_file.getboolean(MODEL_CONFIGURATION_HEADER,
                                                             'is_evaluating')
        except Exception as e:
            print('unable to load if it should be evaluating')

    def create_model_hash(self):

        # BUF_SIZE is totally arbitrary, change for your app!
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

        md5 = hashlib.md5()
        with open(self.model_file + '.data-00000-of-00001', 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)

        return int(md5.hexdigest(), 16) % 2 ** 64
