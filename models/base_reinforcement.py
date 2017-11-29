from models import base_model
import numpy as np
import tensorflow as tf

class BaseReinforcment(base_model.BaseModel):
    """"
    This is the actor critic model.
    """

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 init_exp=0.1,  # initial exploration prob
                 final_exp=0.0,  # final exploration prob
                 anneal_steps=1000,  # N steps for annealing exploration
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, action_handler, is_training,
                         optimizer, summary_writer, summary_every)

        # counters
        self.train_iteration = 0

        # rollout buffer
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

        #training parameters
        self.discount_factor = discount_factor

        # exploration parameters
        self.exploration = init_exp
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps

        self.create_reinforcement_training_model()

    def create_copy_training_model(self, batch_size):
        self.labels = tf.placeholder(tf.int64, shape=(None, self.num_actions))

        cross_entropy = self.action_handler.get_cross_entropy_with_logits(
            labels=self.labels, logits=self.logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        self.train_op = self.optimizer.minimize(loss)

        return loss, self.input, self.labels

    def create_reinforcement_training_model(self):
        """
        Creates a model used for training a bot that will learn through reinforcement
        """
        # this does not create a real valid model
        self.train_op = self.no_op

    def _create_variables(self):
        super()._create_variables()
        # reinforcement variables
        with tf.name_scope("compute_pg_gradients"):
            if self.action_handler.is_split_mode():
                self.taken_actions = tf.placeholder(tf.int32, (None, 4, ), name="taken_actions")
            else:
                self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

        self.no_op = tf.no_op()

    def store_rollout(self, input_state, last_action, reward):
        self.action_buffer.append(last_action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(input_state)

    def update_model(self):
        N = len(self.reward_buffer)
        r = 0  # use discounted reward to approximate Q value

        if N == 0:
            return

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self.reward_buffer[t] + self.discount_factor * r
            discounted_rewards[t] = r

        # whether to calculate summaries
        calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0

        # update policy network with the rollout in batches

        # prepare inputs
        # input_states = self.state_buffer[t][np.newaxis, :]
        # actions = np.array([self.action_buffer[t]])
        # rewards = np.array([discounted_rewards[t]])

        input_states = np.array(self.state_buffer)
        actions = np.array(self.action_buffer)
        rewards = np.array(discounted_rewards)

        # perform one update of training
        result, summary_str = self.sess.run([
            self.train_op,
            self.summarize if calculate_summaries else self.no_op
        ], feed_dict={
            self.input: input_states,
            self.taken_actions: actions,
            self.discounted_rewards: rewards
        })

        # emit summaries
        if calculate_summaries:
            self.summary_writer.add_summary(summary_str, self.train_iteration)

        self.anneal_exploration()
        self.train_iteration += 1

        #print(self.train_iteration)

        # clean up
        self.clean_up()

    def anneal_exploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def clean_up(self):
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

    def reset_model(self):
        self.clean_up()
        self.train_iteration = 0
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_model_name(self):
        return 'base_reinforcement'
