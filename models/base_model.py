import hashlib
import os
import tensorflow as tf

MODEL_CONFIGURATION_HEADER = 'Model Configuration'


class BaseModel:

    config_file = None
    is_initialized = False
    model_file = None
    is_evaluating = False
    no_op = tf.no_op()

    """"
    This is a base class for all models It has a couple helper methods but is mainly used to provide a standard
    interface for running and training a model
    """
    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100):

        # tensorflow machinery
        self.optimizer = optimizer
        self.sess = session
        self.summary_writer = summary_writer

        # debug parameters
        self.summary_every = summary_every

        # for interfacing with the rest of the world
        self.action_handler = action_handler

        # output space
        self.num_actions = num_actions

        # input space
        self.state_dim = state_dim

        self.is_training = is_training

        if self.config_file is not None:
            self.load_config_file()

        # create variables
        self._create_variables()

        # create model
        self.model, self.logits = self.create_model(self.input)

        self.saver = tf.train.Saver()

    def _create_variables(self):
        with tf.name_scope("model_inputs"):
            self.input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state_input")

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

    def sample_action(self, input_state):
        """
        Runs the model to get a single action that can be returned.
        :param input_state: This is the current state of the model at this point in time.
        :return:
        A sample action that can then be used to get controller output.
        """
        #always return an integer
        return 10

    def create_copy_training_model(self, batch_size):
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

    def create_model(self, input):
        """
        Called to create the model, this is called in the constructor
        :param input:
            A placeholder for the input data into the model.
        :return:
            A tensorflow object representing the output of the model
            This output should be able to be run and create an action
            And a tensorflow object representing the logits of the model
            This output should be able to be used in training
        """
        return None, None

    def initialize_model(self):
        """
        This is used to initialize the model variables
        This will also try to load an existing model if it exists
        """
        init = tf.report_uninitialized_variables(tf.global_variables())
        self.sess.run(init)
        model_file = None

        #file does not exist too lazy to add check
        if self.model_file is None:
            model_file = self.get_model_path(self.get_default_file_name() + '.ckpt')
        else:
            model_file = self.model_file
        print(model_file)
        if os.path.isfile(model_file + '.meta'):
            print('loading existing model')
            try:
                self.saver.restore(self.sess, os.path.abspath(model_file))
            except Exception as e:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                print("Unexpected error loading model:", e)
                print('unable to load model')
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print('unable to find model to load')

        self._add_summary_writer()
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
        return dir_path + "\\training\\data\\" + self.get_model_name() + "\\" + filename

    def _add_summary_writer(self):
        if self.summary_writer is not None:
            self.summarize = tf.summary.merge_all()
            # graph was not available when journalist was created
            self.summary_writer.add_graph(self.sess.graph)
        else:
            self.summarize = self.no_op

    def load_config_file(self):
        try:
            self.model_file = self.config_file.get(MODEL_CONFIGURATION_HEADER, 'model_directory')
        except Exception as e:
            print('model directory is not in config', e)

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
