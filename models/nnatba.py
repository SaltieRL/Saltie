import tensorflow as tf



class EmptyModelExample:

    num_hidden_1 = 10 # 1st layer num features



    def create_weights(self):
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.state_dim, self.num_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_actions])),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.num_actions])),
        }

    def encoder(self, input):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.reshape(x, [1, self.state_dim]), self.weights['h1']), self.biases['b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['out']), self.biases['out']))
        return layer_2

    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session,
                 num_actions,
                 state_dim,
                 summary_writer=None,
                 summary_every=100):

        self.saver = tf.train.Saver()
        self.saver.restore(session, "./trained_variables.ckpt")
        self.sess = session
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.input = tf.placeholder(tf.float32, shape=(1, 32))
        self.model = self.create_model()

        init = tf.global_variables_initializer()
        session.run(init)

    def create_model(self):
        self.create_weights()
        hidden_layers = self.encoder(self.input)
        result = tf.argmax(hidden_layers, 1)
        return result

    def store_rollout(self, state, last_action, reward):
        #this is where you apply the rewrad an punioshment

        pass

    def sample_action(self, states):
        return self.sess.run(self.model, feed_dict={self.input: states})

