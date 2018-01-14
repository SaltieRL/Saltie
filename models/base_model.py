import hashlib
import os
import tensorflow as tf
import numpy as np

from conversions.input.input_formatter import InputFormatter
from modelHelpers import tensorflow_feature_creator
from modelHelpers.data_normalizer import DataNormalizer

MODEL_CONFIGURATION_HEADER = 'Model Configuration'


class BaseModel:
    savers_map = {}
    batch_size = 20000
    mini_batch_size = 500
    config_file = None
    is_initialized = False
    model_file = None
    is_evaluating = False
    is_online_training = False
    no_op = tf.no_op()
    train_op = no_op
    logits = None
    is_normalizing = True
    normalizer = None
    feature_creator = None
    load_from_checkpoints = None
    QUICK_SAVE_KEY = 'quick_save'
    network_size = 128
    controller_predictions = None
    input_formatter = None

    """"
    This is a base class for all models It has a couple helper methods but is mainly used to provide a standard
    interface for running and training a model
    """
    def __init__(self, session, num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):

        # tensorflow machinery
        self.optimizer = optimizer
        self.sess = session
        self.summary_writer = summary_writer

        # debug parameters
        self.summary_every = summary_every

        # for interfacing with the rest of the world
        self.action_handler = action_handler

        # player index used for live graphing
        self.player_index = player_index

        # output space
        self.num_actions = num_actions
        self.add_input_formatter(input_formatter_info[0], input_formatter_info[1])
        # input space
        self.state_dim = self.input_formatter.get_state_dim()
        self.state_feature_dim = self.state_dim

        self.is_training = is_training

        if config_file is not None:
            self.config_file = config_file
            self.load_config_file()

        # create variables
        self.stored_variables = self._create_variables()

    def printParameters(self):
        """Visually displays all the model parameters"""
        print('model parameters:')
        print('batch size:', self.batch_size)
        print('mini batch size:', self.mini_batch_size)
        print('using features', (self.feature_creator is not None))

    def _create_variables(self):
        """Creates any variables needed by this model.
        Variables keep their value across multiple runs"""
        with tf.name_scope("model_inputs"):
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state_input")
            self.input = self.input_placeholder
        return {}

    def store_rollout(self, input_state, last_action, reward):
        """
        Used for reinforcment learning this is used to store the last action taken, its state at that point
        and the reward for that action.
        :param input_state: The input state array
        :param last_action: The last action that the bot performed
        :param reward: The reward for performing that action
        :return:
        """
        print(' i do nothing!')

    def store_rollout_batch(self, input_states, last_actions):
        """
        Used for reinforcment learning this is used to store the last action taken, its state at that point
        and the reward for that action.
        :param input_state: The input state array can contain multiple states
        :param last_action: The last set of actions that the bot performed
        :return:
        """
        print(' i do nothing!')

    def add_input_formatter(self, team, index):
        """Creates and adds an input formatter"""
        self.input_formatter = InputFormatter(team, index)

    def create_input_array(self, game_tick_packet, frame_time):
        """Creates the input array from the game_tick_packet"""
        return self.input_formatter.create_input_array(game_tick_packet, frame_time)

    def sample_action(self, input_state):
        """
        Runs the model to get a single action that can be returned.
        :param input_state: This is the current state of the model at this point in time.
        :return:
        A sample action that can then be used to get controller output.
        """
        #always return an integer
        return self.sess.run(self.controller_predictions, feed_dict={self.get_input_placeholder(): input_state})

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        """
        Creates a model used for training a bot that will copy the labeled data

        :param batch_size: The number of batches per a run of the model.
        :return:
            a loss function
            a placeholder for input states
            a placeholder for labels
        """

        #return a loss function in tensorflow
        loss = None
        #return a placeholder for input data
        input = None
        #return a placeholder for labeled data
        labels = None
        return loss, input, labels

    def apply_feature_creation(self, feature_creator):
        self.state_feature_dim = self.state_dim + tensorflow_feature_creator.get_feature_dim()
        self.feature_creator = feature_creator

    def get_input(self, model_input=None):
        """
        Gets the input for the model.
        Also applies normalization
        And feature creation
        :param model_input: input to be used if another input is not None
        :return:
        """
        # use default in case
        if model_input is None:
            safe_input = self.input_placeholder
        else:
            safe_input = model_input

        safe_input = tf.check_numerics(safe_input, 'game tick packet data')

        if self.feature_creator is not None:
            safe_input = self.feature_creator.apply_features(safe_input)

        if self.is_normalizing:
            if self.normalizer is None:
                self.normalizer = DataNormalizer(self.mini_batch_size, self.feature_creator)
            safe_input = self.normalizer.apply_normalization(safe_input)

        return safe_input

    def create_model(self, model_input=None):
        """
        Called to create the model, this is called in the constructor
        :param model_input:
            A Tensor for the input data into the model.
            if None then a default input is used instead
        :return:
            A tensorflow object representing the output of the model
            This output should be able to be run and create an action
            And a tensorflow object representing the logits of the model
            This output should be able to be used in training
        """
        input = self.get_input(model_input)

        self.controller_predictions = self._create_model(input)

    def _create_model(self, model_input):
        """
        Called to create the model, this is not called in the constructor.
        :param model_input:
            A placeholder for the input data into the model.
        :return:
            A tensorflow object representing the output of the model
            This output should be able to be run and create an action that is parsed by the action handler
        """
        return None

    def _initialize_variables(self):
        """
        Initializes all variables attempts to run them with placeholders if those are required
        """
        try:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        except Exception as e:
            print('failed to initialize')
            print(e)
            try:
                init = tf.global_variables_initializer()
                self.sess.run(init, feed_dict={self.input_placeholder: np.zeros((self.batch_size, self.state_dim))})
            except Exception as e2:
                print('failed to initialize again')
                print(e2)
                init = tf.global_variables_initializer()
                self.sess.run(init, feed_dict={
                    self.input_placeholder: np.reshape(np.zeros(206), [1, 206])
                })

    def initialize_model(self):
        """
        This is used to initialize the model variables
        This will also try to load an existing model if it exists
        """
        self._initialize_variables()

        #file does not exist too lazy to add check
        if self.model_file is None:
            model_file = self.get_model_path(self.get_default_file_name())
            self.model_file = model_file
        else:
            model_file = self.model_file
        print('looking for ' + model_file + '.keys')
        if os.path.isfile(model_file + '.keys'):
            print('loading existing model')
            try:
                file = os.path.abspath(model_file)
                self.load_model(os.path.dirname(file), os.path.basename(file))
            except Exception as e:
                self._initialize_variables()
                print("Unexpected error loading model:", e)
                print('unable to load model')
        else:
            self._initialize_variables()
            print('unable to find model to load')

        self._add_summary_writer()
        self.is_initialized = True

    def run_train_step(self, calculate_summaries, input_states, actions):
        """
        Runs a single train step of the model
        :param calculate_summaries: If the model should calculate summaries
        :param input_states: A batch of input states which should equal batch size
        :param actions: A batch of actions which should equal batch size
        :return:
        """
        pass

    def get_model_name(self):
        """
        :return: The name of the model used for saving the file
        """
        return 'base_model'

    def get_default_file_name(self):
        return 'trained_variables'

    def get_model_path(self, filename):
        """
        Creates a path for saving a file, this puts it in the directory of [get_model_name]
        :param filename: name of the file being saved
        :return: The path of the file
        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        return dir_path + "/training/data/" + self.get_model_name() + "/" + filename

    def get_event_path(self, filename, is_replay=False):
        """
        Creates a path for saving tensorflow events for tensorboard, this puts it in the directory of [get_model_name]
        :param filename: name of the file being saved
        :param is_replay: True if the events should be saved for replay analysis
        :return: The path of the file
        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        base_path = "/training/data/events/"
        if is_replay:
            base_path = "/training/replay_events/"
        complete_path = dir_path + base_path + self.get_model_name() + "/" + filename
        modified_path = complete_path
        counter = 0
        while os.path.exists(modified_path):
            counter += 1
            modified_path = complete_path + str(counter)
        return modified_path

    def _add_summary_writer(self):
        """Called to add summary data"""
        if self.summary_writer is not None:
            self.summarize = tf.summary.merge_all()
            # graph was not available when journalist was created
            self.summary_writer.add_graph(self.sess.graph)
        else:
            self.summarize = self.no_op

    def load_config_file(self):
        """Loads a config file.  The config file is stored in self.config_file"""
        try:
            self.model_file = self.config_file.get(MODEL_CONFIGURATION_HEADER, 'model_directory')
        except Exception as e:
            print('model directory is not in config', e)

        try:
            self.batch_size = self.config_file.getint(MODEL_CONFIGURATION_HEADER, 'batch_size')
        except Exception:
            print('batch size is not in config')

        try:
            self.mini_batch_size = self.config_file.getint(MODEL_CONFIGURATION_HEADER, 'mini_batch_size')
        except Exception:
            print('mini batch size is not in config')

        try:
            self.is_evaluating = self.config_file.getboolean(MODEL_CONFIGURATION_HEADER,
                                                             'is_evaluating')
        except Exception as e:
            print('unable to load if it should be evaluating')

        try:
            self.is_normalizing = self.config_file.getboolean(MODEL_CONFIGURATION_HEADER,
                                                              'is_normalizing')
        except Exception as e:
            print('unable to load if it should be normalizing defaulting to true')

    def add_saver(self, name, variable_list):
        """
        Adds a saver to the saver map.
        All subclasses should still use severs_map even if they do not store a tensorflow saver
        :param name: The key of the saver
        :param variable_list: The list of variables to save
        :return: None
        """
        if len(variable_list) == 0:
            print('no variables for saver ', name)
            return
        try:
            self.savers_map[name] = tf.train.Saver(variable_list)
        except Exception as e:
            print('error for saver ', name)
            raise e

    def create_savers(self):
        """Called to create the savers for the model. Or any other way to store the model
        This is called after the model has been created but before it has been initialized.
        This should make calls to add_saver"""
        self.add_saver(self.QUICK_SAVE_KEY, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def _create_model_directory(self, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def _save_keyed_model(self, model_path, key, global_step):
        """
        :param model_path: The directory for which the model should live
        :param key: The key for the savers_map variables
        :param global_step: Which number step in training it is
        """
        keyed_path = self._create_saved_model_path(os.path.dirname(model_path), os.path.basename(model_path), key)
        self._create_model_directory(keyed_path)
        self._save_model(self.sess, self.savers_map[key], keyed_path, global_step)

    def _save_model(self, session, saver, file_path, global_step):
        """
        Saves the model with the specific path, saver, and tensorflow session.
        :param session: The tensorflow session
        :param saver: The object that is actually saving the model
        :param file_path: The place where the model is stored
        :param global_step: What number it is in the training
        """
        try:
            saver.save(session, file_path, global_step=global_step)
        except Exception as e:
            print(e)

    def _create_saved_model_path(self, model_path, file_name, key):
        return model_path + '/' + key + '/' + file_name

    def save_model(self, model_path=None, global_step=None, quick_save=False):
        if model_path is None:
            # use default values
            model_path = self.get_model_path(self.get_default_file_name())
        self._create_model_directory(model_path)
        print('saving model at:\n', model_path)
        file_object = open(model_path + '.keys', 'w')
        if quick_save:
            self._save_keyed_model(model_path, self.QUICK_SAVE_KEY, global_step)
            return

        for key in self.savers_map:
            if key == self.QUICK_SAVE_KEY:
                continue
            file_object.write(key)
            self._save_keyed_model(model_path, key, global_step)
        file_object.close()

    def load_model(self, model_path, file_name, quick_save=False):
        # TODO read keys
        if quick_save:
            self._load_keyed_model(model_path, file_name, self.QUICK_SAVE_KEY)
        print('loading model comprised of', len(self.savers_map))
        for key in self.savers_map:
            if key == self.QUICK_SAVE_KEY:
                continue
            self._load_keyed_model(model_path, file_name, key)

    def _load_keyed_model(self, model_path, file_name, key):
        """
        Loads a model based on a key and a model path
        :param model_path: The base directory of where the model lives
        :param file_name: The name of this specific model piece
        :param key: The key used for the savers_map
        """
        try:
            self._load_model(self.sess, self.savers_map[key], self._create_saved_model_path(model_path, file_name, key))
        except Exception as e:
            print('failed to load model', key)
            print(e)

    def _load_model(self, session, saver, path):
        """
        Loads a model only loads it if the directory exists
        :param session: Tensorflow session
        :param saver: The object that saves and loads the model
        :param path:
        :return:
        """
        if os.path.exists(os.path.dirname(path)):
            checkpoint_path = path
            if self.load_from_checkpoints:
                checkpoint_path = tf.train.load_checkpoint(os.path.dirname(path))
            saver.restore(session, checkpoint_path)
        else:
            print('model for saver not found:', path)

    def create_model_hash(self):
        """Creates the hash of the model used for the server keeping track of what is being used"""
        all_saved_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print(len(all_saved_variables))
        for i in all_saved_variables:
            print(self.player_index, i.name)
        saved_variables = self.sess.run(all_saved_variables)
        saved_variables = np.array(saved_variables)
        return int(hex(hash(str(saved_variables.data))), 16) % 2 ** 64

    def get_input_placeholder(self):
        """Returns the placeholder for getting inputs"""
        return self.input_placeholder

    def get_labels_placeholder(self):
        """Returns the placeholder for getting what actions have been taken"""
        return None
