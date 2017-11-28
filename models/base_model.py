import tensorflow as tf


class BaseModel:

    is_initialized = False

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

        # create variables
        self._create_variables()

        # create model
        self.model = self.create_model(self.input)

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
        """
        return None

    def initialize_model(self):
        """
        This is used to initialize the model variables
        This will also try to load an existing model if it exists
        """
        init = tf.global_variables_initializer()
        self.sess.run(init)

        #file does not exist too lazy to add check

        model_file = self.get_model_path("trained_variables.ckpt")
        print(model_file)
        if os.path.isfile(model_file + '.meta'):
            print('loading existing model')
            try:
                self.saver.restore(self.sess, model_file)
            except:
                print('unable to load model')
        else:
            print('unable to load model')

        self._add_summary_writer()
        self.is_initialized = True

    def get_model_name(self):
        """
        :return: The name of the model used for saving the file
        """
        return 'base_model'

    def get_model_path(self, filename):
        """
        Creates a path for saving a file, this puts it in the directory of [get_model_name]
        :param filename: name of the file being saved
        :return: The path of the file
        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        return dir_path + "\\training\\" + self.get_model_name() + "\\" + filename

    def _add_summary_writer(self):
        if self.summary_writer is not None:
            self.summarize = tf.summary.merge_all()
            # graph was not available when journalist was created
            self.summary_writer.add_graph(self.sess.graph)
