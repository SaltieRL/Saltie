import math
import os
import tensorflow as tf
import numpy as np

from bot_code.modelHelpers import tensorflow_feature_creator
from bot_code.modelHelpers.data_normalizer import DataNormalizer


class BaseModel:
    model = None  # The actual neural net.

    savers_map = {}
    total_batch_size = 20000
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
    input_formatter = None
    summarize = no_op
    iterator = None  # The iterator over the input (training) data
    reg_param = 0.001
    should_regulate = None
    batch_size_placeholder = None
    batch_size_variable = None

    """"
    This is a base class for all models.
    This class is agnostic to what is being learned but subclasses may be more specific.
    However, the all models are just try to learn
    a function that takes some input (eg. game state) to some output (eg. car actions).

    @input_dim and @output_dim speficy the shape of the input/output of the function we are trying the model.

    It has a couple helper methods but is mainly used to provide a standard
    interface for running and training a model.
    """
    def __init__(self,
                 session,
                 input_dim=None,
                 output_dim=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None):

        # tensorflow machinery
        self.train_iteration = 0
        self.optimizer = optimizer
        self.sess = session
        self.summary_writer = summary_writer

        # debug parameters
        self.summary_every = summary_every

        assert input_dim
        assert output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # create variables
        self.stored_variables = self._create_variables()

        self.is_training = is_training

        if config_file is not None:
            self.config_file = config_file
            self.load_config_file()

    def printParameters(self):
        """Visually displays all the model parameters"""
        print('model parameters:')
        print('batch size:', self.total_batch_size)
        print('mini batch size:', self.mini_batch_size)
        print('using features', (self.feature_creator is not None))
        print('regulation parameter', self.reg_param)
        print('is regulating parameter', self.should_regulate)

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

    def create_training_op(self, model_input=None, labels=None):
        """
        Creates the training operation that is run by the trainers.
        This will also call #create_model in the process of creation.

        :param model_input:  The input to the model this is optional
        :param labels: This should be used for creating the loss of the model
        """
        safe_input = self.get_input(model_input, raw_input=True)

        if labels is None:
            labels_input = self.get_labels_placeholder()
        else:
            labels_input = labels

        batched_input, batched_labels = self.create_batched_inputs([safe_input, labels_input])
        batched_converted_input = self.get_input(batched_input)
        with tf.name_scope("training_network"):
            predictions, logits = self.create_model(model_input=batched_converted_input, set_to_object=False, modify_input=False)

        train_op = self._create_training_op(predictions, logits, batched_input, batched_labels)
        if train_op is None:
            raise ValueError("_create_training_op can not return None")
        self.set_train_op(train_op)

    def _create_training_op(self, predictions, logits, raw_model_input, labels):
        """
        Called to create a specific training operation for this one model.
        This should be overwritten by subclasses.
        :param predictions: This is the part of the model that can be used externally to produce predictions
        :param logits: The last layer of the model itself, this is typically the layer before argmax is applied.
        :param raw_model_input: This is an unmodified input that can be used for training uses. (it is batched)
        :param labels: These are the labels that can be used to generate loss
        :return: A tensorflow operation that is used in the training step
        """
        raise NotImplementedError('Derived classes must override this.')

    def set_train_op(self, train_operation):
        """
        Sets the train operation for the model
        :param train_operation:
        """
        if train_operation is None:
            raise ValueError("can not set the training operation to a null value")
        self.train_op = train_operation

    def create_batched_inputs(self, inputs):
        """
        Takes in the inputs and creates a batch variation of them.
        :param inputs: This is an array or tuple of inputs that will be converted to their batch form.
        :return: The outputs converted to their batch form.
        """
        outputs = tuple(inputs)
        if self.total_batch_size > self.mini_batch_size:
            ds = tf.data.Dataset.from_tensor_slices(tuple(inputs)).batch(self.mini_batch_size)
            self.iterator = ds.make_initializable_iterator(shared_name="model_iterator")
            outputs = self.iterator.get_next()
            self.batch_size_variable = tf.shape(outputs[0])[0]
        else:
            self.batch_size_variable = self.batch_size_placeholder
        return outputs

    def create_training_feed_dict(self, input_array, label_array):
        return {self.get_input_placeholder(): input_array,
                self.get_labels_placeholder(): label_array,
                self.batch_size_placeholder: [min(self.total_batch_size, self.mini_batch_size)]}

    def run_train_step(self, should_calculate_summaries, feed_dict=None, epoch=-1):
        """
        Runs a single train step of the model.
        If batching is enable this will internally handle batching as well
        :param should_calculate_summaries: True if summaries/logs from this train step should be saved. False otherwise
        :param feed_dict: The inputs we feed into the model.
        :param epoch: What number iteration we should be on
        :return: The epoch number of the internal model state
        """

        if epoch != -1:
            self.train_iteration = epoch

        should_summarize = should_calculate_summaries and self.summarize is not None and self.summary_writer is not None

        # perform one update of training
        if self.total_batch_size > self.mini_batch_size:
            _, = self.sess.run([self.iterator.initializer],
                          feed_dict=feed_dict)
            num_batches = math.ceil(float(self.total_batch_size) / float(self.mini_batch_size))
            # print('num batches', num_batches)
            counter = 0
            while counter < num_batches:
                try:
                    result, summary_str = self.sess.run([
                        self.train_op,
                        self.summarize if should_summarize else self.no_op
                    ])
                    # emit summaries
                    if should_summarize:
                        self.summary_writer.add_summary(summary_str, self.train_iteration)
                        self.train_iteration += 1
                    counter += 1
                except tf.errors.OutOfRangeError:
                    break
                except Exception as e:
                    print(e)
            print('batch amount:', counter)
        else:
            result, summary_str = self.sess.run([
                    self.train_op,
                    self.summarize if should_summarize else self.no_op
                ],
                feed_dict=feed_dict)
            # emit summaries
            if should_summarize:
                self.summary_writer.add_summary(summary_str, self.train_iteration)
                self.train_iteration += 1
        return self.train_iteration

    def get_input(self, model_input=None, raw_input=False):
        """
        Gets the input for the model.
        Also applies normalization
        And feature creation
        :param model_input: input to be used if another input is not None
        :param raw_input: used in some cases this will not perform feature creation or normalization
        :return: An input that is not None and will apply transformations based on the configs
        """
        # use default in case
        if model_input is None:
            safe_input = self.input_placeholder
        else:
            safe_input = model_input

        safe_input = tf.check_numerics(safe_input, 'game tick packet data')

        if raw_input:
            return safe_input

        if self.feature_creator is not None:
            safe_input = self.feature_creator.apply_features(safe_input)

        if self.is_normalizing:
            if self.normalizer is None:
                self.normalizer = DataNormalizer(self.mini_batch_size, self.feature_creator)
            safe_input = self.normalizer.apply_normalization(safe_input)

        return safe_input

    def create_model(self, model_input=None, set_to_object=True, modify_input=True):
        """
        Called to create the model, this is called in the constructor
        :param model_input:
            A Tensor for the input data into the model.
            if None then a default input is used instead
        :param set_to_object:
            This is true if the values should be internally set
            If true #model and #logits will be set to the values returned in this method
        :param modify_input:
            If true this will modify the input based on other configs.
            If false model_input is passed straight to the subclasses even if it is null.
        :return:
            A tensorflow object representing the output of the model
            This output should be able to be run and create a prediction. (Typically argmax)
            A second tensorflow object representing the raw output layer (Run before argmax)
        """
        if modify_input:
            input = self.get_input(model_input)
        else:
            input = model_input

        model, logits = self._create_model(input, self.get_batch_size())
        if logits is None:
            raise NotImplementedError("Logits must be set after create model is called")
        if set_to_object:
            self.model = model
            self.logits = logits
        return model, logits

    def _create_model(self, model_input, batch_size):
        """
        Called to create the model, this is not called in the constructor.
        :param model_input:
            A placeholder for the input data into the model.
        :param batch_size:
            The batch size for this run as a tensor
        :return:
            A tensorflow object representing the output of the model
            This output should be able to be run and create an action that is parsed by the action handler
        """
        raise NotImplementedError('Derived classes must override this.')

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
                self.sess.run(init, feed_dict={self.input_placeholder: np.zeros((self.total_batch_size, self.input_dim))})
            except Exception as e2:
                print('failed to initialize again')
                print(e2)
                init = tf.global_variables_initializer()
                self.sess.run(init, feed_dict={
                    # TODO: This seems sepcific to agents. Refactor this out of base_model.py
                    self.input_placeholder: np.reshape(np.zeros(206), [1, 206])
                })

    def initialize_model(self):
        """
        This is used to initialize the model variables
        This will also try to load an existing model if it exists
        """
        self._initialize_variables()

        if self.model_file is None or not os.path.isfile(self.model_file):
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
            print('unable to find model to load')

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.sess.graph)
            self.summarize = tf.summary.merge_all()
        self.is_initialized = True

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
        return os.path.join(dir_path, "training", "saltie", self.get_model_name(), filename)

    def get_event_path(self, filename, is_replay=False):
        """
        Creates a path for saving tensorflow events for tensorboard, this puts it in the directory of [get_model_name]
        :param filename: name of the file being saved
        :param is_replay: True if the events should be saved for replay analysis
        :return: The path of the file
        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        base_path = os.path.join("training", "training_events")
        if is_replay:
            base_path = os.path.join("training", "in_game_events")
        complete_path = os.path.join(dir_path, base_path, self.get_model_name(), filename)
        modified_path = complete_path
        counter = 0
        while os.path.exists(modified_path):
            counter += 1
            modified_path = complete_path + str(counter)
        return modified_path

    def add_summary_writer(self, event_name, is_replay=False):
        """
        Called to add a way to summarize the model info.
        This could be called before the graph is finalized
        :param event_name: The file name of the summary
        :param is_replay: True if the events should be saved for replay analysis
        :return:
        """
        self.summary_writer = tf.summary.FileWriter(self.get_event_path(event_name, is_replay))

    def load_config_file(self):
        """Loads a config file.  The config file is stored in self.config_file"""
        try:
            self.model_file = self.config_file.get('model_directory', self.model_file)
        except Exception as e:
            print('model directory is not in config', e)

        try:
            self.total_batch_size = self.config_file.getint('batch_size', self.total_batch_size)
        except Exception:
            print('batch size is not in config')

        try:
            self.mini_batch_size = self.config_file.getint('mini_batch_size', self.mini_batch_size)
        except Exception:
            print('mini batch size is not in config')

        try:
            self.is_evaluating = self.config_file.getboolean('is_evaluating', self.is_evaluating)
        except Exception as e:
            print('unable to load if it should be evaluating')

        try:
            self.is_normalizing = self.config_file.getboolean('is_normalizing', self.is_normalizing)
        except Exception as e:
            print('unable to load if it should be normalizing defaulting to true')
        try:
            self.should_regulate = self.config_file.getboolean('should_regulate', True)
        except Exception as e:
            print('unable to load if it should be regulating defaulting to true')
        try:
            self.reg_param = self.config_file.getfloat('regulate_param', self.reg_param)
        except Exception as e:
            print('unable to load if it should be regulating defaulting to true')
        print('done loading')

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
        return os.path.join(model_path, key, file_name)

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

    def _create_variables(self):
        """Creates any variables needed by this model.
        Variables keep their value across multiple runs"""
        with tf.name_scope("model_inputs"):
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="inputs")
            self.batch_size_placeholder = tf.placeholder(tf.int32, shape=[1])
            self.batch_size_variable = self.batch_size_placeholder

        return {}

    def get_labels_placeholder(self):
        """Returns the placeholder for getting what actions have been taken"""
        return self.no_op

    def get_batch_size(self):
        """
        Returns the batch size as a tensor.
        """
        return self.batch_size_variable

    def get_variables_activations(self):
        """
        Returns the weights, biases and activation type for each layer
        :return: Return using [layer1, layer2, etc.] layer: [weights, biases, activation]
        weights: [neuron0, neuron1, neuron2, etc.] which each include (from prev. layer): [neuron0, neuron1, etc.]
        biases: [neuron0, neuron1, etc.] Each holding the bias value.
        ex. layer: [[[[1, 2, 3], [2, 5, 1], [2, 5, 1]], [1, 4, 2, 1, 4], 'relu'], next layer]
        """
        r = list()
        weights = list()
        biases = list()
        for i in range(7):
            biases.append(np.random.randint(-10, 10))
        r.append([[], biases, 'relu'])
        biases.clear()
        for i in range(5):
            temp = list()
            for n in range(7):
                temp.append(np.random.randint(-20, 20))
            weights.append(temp)
            biases.append(np.random.rand())
        r.append([weights, biases, 'sigmoid'])
        return r

    def get_activations(self, input_array=None):
        return [[np.random.randint(0, 30) for i in range(7)], [np.random.rand() for i in range(5)]]

    def get_regularization_loss(self, variables, prefix=None):
        """Gets the regularization loss from the varaibles.  Used if the weights are getting to big"""
        normalized_variables = [tf.reduce_sum(tf.nn.l2_loss(x) * self.reg_param)
                                for x in variables]

        reg_loss = tf.reduce_sum(normalized_variables, name=(prefix + '_reg_loss'))
        tf.summary.scalar(prefix + '_reg_loss', reg_loss)
        if self.should_regulate:
            return reg_loss * (self.reg_param * 10.0)
        else:
            return tf.constant(0.0)
