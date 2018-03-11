from bot_code.models.base_agent_model import BaseAgentModel
import tensorflow as tf


class NNAtba(BaseAgentModel):

    keep_prob = 0.5
    num_hidden_1 = 500 # 1st layer num features
    num_hidden_2 = 1000 # 1st layer num features
    labels = None
    weights = None
    biases = None
    logits = None

    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def _create_model(self, model_input):
        self.create_weights()
        logits = self.encoder(model_input)
        return self.action_handler.create_model_output(logits), logits

    def create_weights(self):
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.state_feature_dim, self.num_hidden_1]), name='wh1'),
            'h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2]), name='wh2'),
            'out': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_actions]), name='wout'),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden_1]), name='bh1'),
            'b2': tf.Variable(tf.random_normal([self.num_hidden_2]), name='bh2'),
            'out': tf.Variable(tf.random_normal([self.num_actions]), name='bout'),
        }

    def encoder(self, input):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu6(tf.add(tf.matmul(input, self.weights['h1']), self.biases['b1']))

        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

        if (self.is_training):
            layer_2 = tf.nn.dropout(layer_2, self.keep_prob)

        # Encoder Hidden layer with sigmoid activation #3
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out']))
        return layer_3

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        self.labels = tf.placeholder(tf.int64, shape=(None, self.num_actions))

        cross_entropy = self.action_handler.get_action_loss_from_logits(
            labels=self.labels, logits=self.logits, index=0)
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        self.train_op = self.optimizer.minimize(loss)

        return loss, self.input, self.labels

    def sample_action(self, input_state):
        return self.sess.run(self.model, feed_dict={self.input: input_state})[0]

    def get_model_name(self):
        return 'nnatba' + ('_split' if self.action_handler.is_split_mode else '')
