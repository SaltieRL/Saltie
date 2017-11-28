import tensorflow as tf
import os



class ActorCritic:

    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100):

        self.action_handler = action_handler
        self.is_training = is_training
        self.optimizer = optimizer
        self.sess = session
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.input = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.model = self.create_model(self.input)

        self.saver = tf.train.Saver()

    def initialize_model(self):
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

    def create_training_model_copy(self, batch_size):
        self.labels = tf.placeholder(tf.int64, shape=(None, self.num_actions))

        cross_entropy = self.action_handler.get_cross_entropy_with_logits(
            tf, labels=self.labels, logits=self.logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        optimizer = self.optimizer.minimize(loss)

        return loss, self.input, self.labels, optimizer

    def create_model(self, input):
        self.create_weights()
        self.logits = self.encoder(input)
        return self.action_handler.create_model_output(tf, self.logits)

    def store_rollout(self, state, last_action, reward):
        #I only care about the current state and action
        pass

    def sample_action(self, states):
        return self.sess.run(self.model, feed_dict={self.input: states})[0]

    def get_model_path(self, filename):
        # go up two levels
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        return dir_path + "\\training\\data\\nnatba\\" + filename

