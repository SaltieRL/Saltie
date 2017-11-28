import tensorflow as tf

from models.atbas import nnatba

CUDNN = "cudnn"
BASIC = "basic"

class RNNAtba(nnatba.NNAtba):

    num_hidden_1 = 500 # 1st layer num features
    labels = None
    num_layers = 5
    hidden_size = 300
    rnn_mode = CUDNN
    num_steps = 10
    batch_size = 1
    keep_prob = 0.5
    init_scale = .99

    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=None,
                 summary_writer=None,
                 summary_every=100):

        super().__init__(session, state_dim, num_actions, action_handler, is_training,
                         optimizer, summary_writer, summary_every)

    def create_copy_training_model(self, batch_size):
        self.batch_size = batch_size
        return super().create_copy_training_model(batch_size)

    def create_model(self, input):
        self.create_weights()
        hidden_layers = self.input_encoder(input)
        hidden_layers = tf.expand_dims(hidden_layers, 0)
        output, last_state = self._build_rnn_graph(inputs=hidden_layers, config=None, is_training=self.is_training)
        with tf.variable_scope('RNN'):
            output_w = tf.get_variable('output_w', [self.hidden_size, output.get_shape()[1]])
            output_b = tf.get_variable('output_b', [output.get_shape()[1]])

        output = tf.reshape(output, [-1, self.hidden_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)

        self.logits = self.rnn_decoder(output)
        return self.action_handler.create_model_output(self.logits)

    def create_weights(self):
        self.weights = {
            'inputW': tf.Variable(tf.random_normal([self.state_dim, self.hidden_size]), name='inputW'),
            'rnn_outputW': tf.Variable(tf.random_normal([self.hidden_size, self.num_hidden_1]), name='rnn_outputW'),
            'out': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_actions]), name='outputW'),
        }
        self.biases = {
            'inputB': tf.Variable(tf.random_normal([self.hidden_size]), name='inputB'),
            'rnn_outputB': tf.Variable(tf.random_normal([self.num_hidden_1]), name='rnn_outputB'),
            'out': tf.Variable(tf.random_normal([self.num_actions]), name='outputB'),
        }

    def input_encoder(self, input):
        # Encoder Hidden layer with sigmoid activation #1
        inputs = tf.nn.relu(tf.add(tf.matmul(input, self.weights['inputW']), self.biases['inputB']), name='input_layer')
        return inputs

    def rnn_decoder(self, rnn_out):
        # Encoder Hidden layer with sigmoid activation #2
        hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(rnn_out, self.weights['rnn_outputW']), self.biases['rnn_outputB']), name='rnn_out')

        # Encoder Hidden layer with sigmoid activation #3
        return tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, self.weights['out']), self.biases['out']), name='logits')

    def _build_rnn_graph(self, inputs, config, is_training):
        if self.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self.num_layers,
            num_units=self.hidden_size,
            input_size=self.hidden_size,
            dropout=1 - self.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -self.init_scale, self.init_scale),
            validate_shape=False)
        c = tf.zeros([self.num_layers, self.batch_size, self.hidden_size],
                     tf.float32)
        h = tf.zeros([self.num_layers, self.batch_size, self.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if self.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if self.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                self.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % self.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and self.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=self.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size])
        return output, state

    def get_model_name(self):
        return 'rnnatba' + ('_split' if self.action_handler.is_split_mode else '')
