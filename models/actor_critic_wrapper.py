from models.actorcritic import PolicyGradientActorCritic
import tensorflow as tf

class ActorCriticModel:

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 is_training=False,
                 summary_writer=None,
                 summary_every=100,
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):

        self.session = session
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.pg_reinforce = PolicyGradientActorCritic(session,
                                                      optimizer,
                                                      self.actor_network,
                                                      self.critic_network,
                                                      self.state_dim,
                                                      self.num_actions,
                                                      summary_writer=summary_writer)
    def initialize_model(self):
        pass

    def actor_network(self, states):
        # define policy neural network
        W1 = tf.get_variable("W1", [self.state_dim, 20],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1", [20],
                             initializer=tf.constant_initializer(0))
        h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
        W2 = tf.get_variable("W2", [20, self.num_actions],
                             initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("b2", [self.num_actions],
                             initializer=tf.constant_initializer(0))
        p = tf.matmul(h1, W2) + b2
        return p

    def critic_network(self, states):
        # define policy neural network
        W1 = tf.get_variable("W1", [self.state_dim, 20],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1", [20],
                             initializer=tf.constant_initializer(0))
        h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
        W2 = tf.get_variable("W2", [20, 1],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b2", [1],
                             initializer=tf.constant_initializer(0))
        v = tf.matmul(h1, W2) + b2
        return v

    def store_rollout(self, state, last_action, reward):
        self.pg_reinforce.store_rollout(state, last_action, reward)


    def sample_action(self, states):
        return self.pg_reinforce.sampleAction(states)
