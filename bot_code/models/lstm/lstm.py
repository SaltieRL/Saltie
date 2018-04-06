import numpy as np
import tensorflow as tf

from bot_code.models.base_agent_model import BaseAgentModel


class BaseLSTMModel(BaseAgentModel):
    num_epochs = 100
    num_actions = 8
    total_series_length = 50000
    truncated_backprop_length = 15
    echo_step = 3
    hidden_size = 219

    def __init__(self,
                 session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None):

        self.num_actions = num_actions
        print ('lstm', 'num_actions', num_actions)
        if input_formatter_info is None:
            input_formatter_info = [0, 0]
        super().__init__(session, num_actions,
                         input_formatter_info=input_formatter_info,
                         player_index=player_index,
                         action_handler=action_handler,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

    def _create_model(self, model_input, batch_size):
        self.create_weights()
        input_ = self.input_encoder(model_input)
        input_ = tf.expand_dims(input_, 1)
        # Forward passes
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_dim)
        # defining initial state
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        with tf.variable_scope('recurrent_layer', reuse=tf.AUTO_REUSE):
            output, state = tf.nn.dynamic_rnn(cell, input_, initial_state=initial_state, dtype=tf.float32)
        output = tf.reshape(output, [-1, self.hidden_size])
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            output_w = tf.get_variable('output_w', [self.hidden_size, output.get_shape()[1]])
            output_b = tf.get_variable('output_b', [output.get_shape()[1]])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.logits = self.rnn_decoder(output)
        return self.action_handler.create_model_output(self.logits), self.logits

    def create_weights(self):
        self.weights = {
            'h1': tf.Variable(np.random.rand(self.state_feature_dim, self.hidden_size), dtype=tf.float32),
            'h2': tf.Variable(np.random.rand(self.hidden_size, self.hidden_size), dtype=tf.float32),
            'out': tf.Variable(np.random.rand(self.hidden_size, self.num_actions), dtype=tf.float32),
        }
        self.biases = {
            'b1': tf.Variable(np.zeros((1, self.hidden_size)), dtype=tf.float32),
            'b2': tf.Variable(np.zeros((1, self.hidden_size)), dtype=tf.float32),
            'out': tf.Variable(np.zeros((1, self.num_actions)), dtype=tf.float32)
        }
        self.add_saver('vars',
                       tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def _create_variables(self):
        super()._create_variables()
        self.labels = tf.placeholder(tf.float32,
                                     (None, self.action_handler.get_number_actions()),
                                     name="taken_actions_phd")

    def input_encoder(self, input):
        inputs = tf.nn.relu(tf.add(tf.matmul(input, self.weights['h1']), self.biases['b1']), name='input_layer')
        return inputs

    def rnn_decoder(self, rnn_out):
        # Encoder Hidden layer with sigmoid activation #2
        hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(rnn_out, self.weights['h2']), self.biases['b2']), name='rnn_out')

        # Encoder Hidden layer with sigmoid activation #3
        return tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, self.weights['out']), self.biases['out']), name='logits')

    def _create_split_training_op(self, indexes, logits, labels, *args):
        if len(labels.get_shape()) == 2:
            labels = tf.squeeze(labels, axis=[1])
        cross_entropy = self.action_handler.get_action_loss_from_logits(
            labels=labels, logits=logits, index=indexes)
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean' + str(indexes))

        tf.summary.scalar("loss", tf.reduce_mean(loss))

        return loss

    def _process_results(self, central_result, split_result):
        total_loss = 0.0
        for loss in split_result:
            total_loss += loss

        tf.summary.scalar("total_loss", total_loss)
        return self.optimizer.minimize(total_loss)

    def _create_central_training_op(self, predictions, logits, raw_model_input, labels):
        return None

    def get_model_name(self):
        return 'rnn' + ('_split' if self.action_handler.is_split_mode else '')

    def get_labels_placeholder(self):
        return self.labels
