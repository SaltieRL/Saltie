import numpy as np
import tensorflow as tf

from bot_code.models.base_agent_model import BaseAgentModel


class BaseReinforcement(BaseAgentModel):
    """"
    This is the actor critic model.
    """

    action_threshold = 0.1
    taken_actions = None

    def __init__(self, session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None,
                 init_exp=0.05,  # initial exploration prob
                 final_exp=0.0,  # final exploration prob
                 anneal_steps=1000,  # N steps for annealing exploration
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, num_actions,
                         input_formatter_info=input_formatter_info,
                         player_index=player_index,
                         action_handler=action_handler,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

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

    def printParameters(self):
        super().printParameters()
        print('Reinforcment Parameters:')
        print('discount factor', self.discount_factor)

    def _initialize_variables(self):
        try:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        except Exception as e:
            print('failed to initialize')
            print(e)
            try:
                init = tf.global_variables_initializer()
                if self.action_handler.is_split_mode():
                    actions_null = np.zeros((self.batch_size, self.action_handler.get_number_actions()))
                else:
                    actions_null = np.zeros((self.batch_size,))
                self.sess.run(init, feed_dict={self.get_input_placeholder(): np.zeros((self.batch_size, self.state_dim)),
                                               self.taken_actions_placeholder: actions_null})
            except Exception as e2:
                print('failed to initialize again')
                print(e2)
                init = tf.global_variables_initializer()
                self.sess.run(init, feed_dict={
                    self.input_placeholder: np.reshape(np.zeros(self.state_dim), [1, self.state_dim])
                })

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
                self.taken_actions_placeholder = tf.placeholder(tf.float32,
                                                                (None, self.action_handler.get_number_actions()),
                                                                name="taken_actions_phd")
            else:
                self.taken_actions_placeholder = tf.placeholder(tf.float32, (None,), name="taken_actions_phd")
            self.taken_actions = self.taken_actions_placeholder
            self.input_rewards = self.create_reward()
        return {}

    def get_labels_placeholder(self):
        return self.taken_actions

    def store_rollout(self, input_state, last_action, reward):
        if self.is_training:
            if self.action_buffer is None:
                self.action_buffer = []
                self.state_buffer = []
                self.reward_buffer = []
            self.action_buffer.append(last_action)
            self.reward_buffer.append(reward)
            self.state_buffer.append(input_state)

        if len(self.action_buffer) >= self.batch_size and self.is_online_training and not self.is_evaluating:
            print('running online trainer!')
            self.update_model()
        if self.action_buffer is not None and len(self.action_buffer) >= self.batch_size * 10:
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

    def discount_rewards(self, input_rewards, input):
        r = tf.Variable(initial_value=tf.reshape(tf.constant(0.0), [1]))
        length = tf.Variable(tf.size(input_rewards))
        discounted_rewards = tf.zeros(tf.shape(input_rewards), name='discounted_rewards')
        counter = tf.Variable(length)
        tf.while_loop(lambda i, _, __, ___: tf.greater_equal(i, 0), self.in_loop,
                      (counter, input_rewards, discounted_rewards, r),
                      parallel_iterations=1, back_prop=False)
        return discounted_rewards

    def update_model(self):
        if len(self.state_buffer) == 0:
            return
        # whether to calculate summaries

        # update policy network with the rollout in batches
        input_states = np.array(self.state_buffer)
        actions = np.array(self.action_buffer)
        rewards = None
        self.run_train_step(True, feed_dict=self.create_feed_dict(input_states, actions))

        self.anneal_exploration()
        self.train_iteration += 1

        # clean up
        self.clean_up()

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
