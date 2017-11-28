from models import base_reinforcement
import tensorflow as tf
import random

class BaseActorCritic(base_reinforcement.BaseReinforcment):

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)

    def create_model(self, input):
        with tf.name_scope("predict_actions"):
            # initialize actor-critic network
            with tf.variable_scope("actor_network"):
                self.policy_outputs = self.actor_network(self.input)
            with tf.variable_scope("critic_network"):
                self.value_outputs = self.critic_network(self.input)

            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            self.predicted_actions = \
                self.action_handler.optionally_split_tensors(tf, self.action_scores,
                                                             lambda split_tensor: tf.multinomial(split_tensor, 1))

        # get variable list
        self.actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
        self.critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

    def create_reinforcement_training_model(self):
        with tf.name_scope("training_network"):
            with tf.variable_scope("actor_network", reuse=True):
                self.logprobs = self.actor_network(self.input)

            with tf.variable_scope("critic_network", reuse=True):
                self.estimated_values = self.critic_network(self.input)

            log_probs_split = \
                self.action_handler.optionally_split_tensors(tf, self.logprobs,
                                                             lambda split_tensor: tf.identity(split_tensor))

                # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log_probs_split,
                                                                                     labels=self.taken_actions)
        self.create_training_op()

    def create_training_op(self):
        self.train_op = self.optimizer.minimize(self.cross_entropy_loss)

    def sample_action(self, input_state):
        # TODO: use this code piece when tf.multinomial gets better
        # sample action from current policy
        # actions = self.session.run(self.predicted_actions, {self.input: input})[0]
        # return actions[0]

        # epsilon-greedy exploration strategy
        if random.random() < self.exploration:
            return random.randint(0, self.num_actions - 1)
        else:
            softmax = self.action_handler.optionally_split_tensors(tf, self.action_scores,
                                                         lambda input_tensor: tf.nn.softmax(input_tensor))
            action_scores = self.sess.run(softmax, {self.input: input_state})[0]

            action = self.action_handler.\
                optionally_split_numpy_arrays(action_scores,
                                              lambda input_array: np.argmax(np.random.multinomial(1, input_array)),
                                              is_already_split=True)
            return action

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

