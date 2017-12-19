import numpy as np
import tensorflow as tf

from models import base_model


class BaseReinforcement(base_model.BaseModel):
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
                 init_exp=0.05,  # initial exploration prob
                 final_exp=0.0,  # final exploration prob
                 anneal_steps=1000,  # N steps for annealing exploration
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, action_handler, is_training,
                         optimizer, summary_writer, summary_every)

        # counters
        self.train_iteration = 0

        # rollout buffer
        self.state_buffer = None
        self.reward_buffer = []
        self.action_buffer = None

        # training parameters
        self.discount_factor = discount_factor

        # exploration parameters
        self.exploration = init_exp
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps

    def _set_variables(self):
        try:
            init = tf.global_variables_initializer()
            if self.action_handler.is_split_mode():
                actions_null = np.zeros((2000, 4))
            else:
                actions_null = np.zeros((2000,))
            self.sess.run(init, feed_dict={self.input_placeholder: np.zeros((2000, 206)), self.taken_actions_placeholder: actions_null})
            print ('done initializing')
        except Exception as e:
            print('failed to initialize')
            print(e)
            try:
                init = tf.global_variables_initializer()
                self.sess.run(init)
            except Exception as e2:
                print('failed to initialize again')
                print(e2)
                init = tf.global_variables_initializer()
                self.sess.run(init, feed_dict={
                    self.input: np.reshape(np.zeros(206), [1, 206])
                })


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
                self.taken_actions_placeholder = tf.placeholder(tf.int32, (2000, 4), name="taken_actions_phd")
                self.taken_actions = tf.Variable(self.taken_actions_placeholder)
            else:
                self.taken_actions_placeholder = tf.placeholder(tf.int32, (2000,), name="taken_actions_phd")
                self.taken_actions = tf.Variable(self.taken_actions_placeholder)
            self.input_rewards = self.create_reward()

        ds = tf.data.Dataset.from_tensor_slices((self.input, self.taken_actions)).batch(100)
        self.iterator = ds.make_initializable_iterator()
        return {}

    def store_rollout(self, input_state, last_action, reward):
        if self.is_training:
            if self.action_buffer is None:
                self.action_buffer = []
                self.state_buffer = []
                self.reward_buffer = []
            self.action_buffer.append(last_action)
            self.reward_buffer.append(reward)
            self.state_buffer.append(input_state)

        if len(self.action_buffer) >= 1000 and self.is_online_training and not self.is_evaluating:
            print('running online trainer!')
            self.update_model()
        if self.action_buffer is not None and len(self.action_buffer) >= 10000:
            self.clean_up()

    def store_rollout_batch(self, input_state, last_action):
        if self.is_training:
            if self.action_buffer is None:
                self.action_buffer = last_action
                self.state_buffer = input_state
            else:
                self.action_buffer = np.concatenate((self.action_buffer, last_action), axis=0)
                self.state_buffer = np.concatenate((self.state_buffer, input_state), axis=0)

    def in_loop(self, counter, input_rewards, discounted_rewards, r):
        new_r = input_rewards[counter] + self.discount_factor * r
        index = tf.reshape(counter, [1])
        update_tensor = tf.scatter_nd(index, new_r, tf.shape(discounted_rewards))
        new_discounted_rewards = discounted_rewards + update_tensor
        new_counter = counter - tf.constant(1)
        return new_counter, input_rewards, new_discounted_rewards, new_r

    def discount_rewards(self, input_rewards):
        r = tf.Variable(initial_value=tf.reshape(tf.constant(0.0), [1]))
        length = tf.Variable(tf.size(input_rewards))
        discounted_rewards = tf.zeros(tf.shape(input_rewards), name='discounted_rewards')
        counter = tf.Variable(length)
        tf.while_loop(lambda i, _, __, ___: tf.greater_equal(i, 0), self.in_loop,
                      (counter, input_rewards, discounted_rewards, r),
                      parallel_iterations=1, back_prop=False)
        return discounted_rewards

    def update_model(self):
        N = len(self.state_buffer)
        r = 0  # use discounted reward to approximate Q value

        if N == 0:
            return

        # compute discounted future rewards
        # discounted_rewards = np.zeros(N)
        # for t in reversed(range(N)):
        #    # future discounted reward from now on
        #    r = self.reward_buffer[t] + self.discount_factor * r
        #    discounted_rewards[t] = r

        # whether to calculate summaries
        calculate_summaries = self.summarize is not None and self.summary_writer is not None and self.train_iteration % self.summary_every == 0

        # update policy network with the rollout in batches

        # prepare inputs
        # input_states = self.state_buffer[t][np.newaxis, :]
        # actions = np.array([self.action_buffer[t]])
        # rewards = np.array([discounted_rewards[t]])

        input_states = np.array(self.state_buffer)
        actions = np.array(self.action_buffer)
        # rewards = np.array(self.reward_buffer).reshape((len(self.reward_buffer), 1))
        rewards = None
        result, summary_str = self.run_train_step(calculate_summaries, input_states, actions, rewards)

        # emit summaries
        if calculate_summaries:
            self.summary_writer.add_summary(summary_str, self.train_iteration)

        self.anneal_exploration()
        self.train_iteration += 1

        # print(self.train_iteration)

        # clean up
        self.clean_up()

    def run_train_step(self, calculate_summaries, input_states, actions, rewards):
        # perform one update of training
        result, summary_str = self.sess.run([
            self.train_op,
            self.summarize if calculate_summaries else self.no_op
        ], feed_dict={
            self.input_placeholder: input_states,
            self.taken_actions_placeholder: actions,
            self.input_rewards: rewards
        })
        return result, summary_str

    def anneal_exploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def clean_up(self):
        self.state_buffer = None
        self.reward_buffer = []
        self.action_buffer = None

    def reset_model(self):
        self.clean_up()
        self.train_iteration = 0
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_model_name(self):
        return 'base_reinforcement'

    def create_reward(self):
        return tf.placeholder(tf.float32, (None, 1), name="input_rewards")
