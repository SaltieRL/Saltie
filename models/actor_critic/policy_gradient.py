import numpy as np
import tensorflow as tf
import math

from models import base_model
from models.actor_critic.base_actor_critic import BaseActorCritic
from modelHelpers import tensorflow_reward_manager


class PolicyGradient(BaseActorCritic):
    max_gradient = 1.0
    total_loss_divider = 1.0

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None,
                 discount_factor=0.99,  # discount future rewards
                 ):
        self.reward_manager = tensorflow_reward_manager.TensorflowRewardManager(state_dim)

        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file, discount_factor)

    def printParameters(self):
        super().printParameters()
        print('policy gradient parameters:')
        print('max gradient allowed:', self.max_gradient)
        print('amount to squash total loss:', self.total_loss_divider)

    def load_config_file(self):
        super().load_config_file()
        try:
            self.max_gradient = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                      'max_gradient')
        except:
            print('unable to load max_gradient')
        try:
            self.max_gradient = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                        'total_loss_divider')
        except:
            print('unable to load total_loss_divider')

    def get_input(self, model_input=None):
        if model_input is None:
            return super().get_input(self.input)
        else:
            return super().get_input(model_input)

    def create_training_op(self, logprobs, taken_actions):
        critic_gradients, critic_loss, critic_reg_loss = self.create_critic_gadients()
        actor_gradients, actor_loss, actor_reg_loss = self.create_actor_gradients(logprobs, taken_actions)

        tf.summary.scalar("total_reg_loss", critic_reg_loss + actor_reg_loss)

        return self._compute_training_op(actor_gradients, critic_gradients)

    def create_advantages(self):
        # compute advantages A(s) = R - V(s)
        return tf.reduce_sum(self.discounted_rewards - self.estimated_values, name='advantages')

    def create_actor_gradients(self, logprobs, taken_actions):
        advantages = self.create_advantages()

        actor_reg_loss = self.get_regularization_loss(self.all_but_last_actor_layer, prefix="actor_hidden")
        indexes = np.arange(0, len(self.action_handler.get_action_sizes()), 1).tolist()

        result = self.action_handler.run_func_on_split_tensors([indexes,
                                                                logprobs,
                                                                taken_actions,
                                                                advantages,
                                                                self.last_row_variables],
                                                               self.create_split_actor_loss,
                                                               return_as_list=True)

        merged_gradient_list = []
        total_loss = 0
        for item in result:
            merged_gradient_list += item[0]
            total_loss += item[1]

        tf.summary.scalar("total_actor_loss", tf.reduce_mean(total_loss))

        total_loss = total_loss / self.total_loss_divider

        # total_loss += actor_reg_loss

        # total_loss = tf.Print(total_loss, [total_loss], 'total_loss')

        total_loss = tf.identity(total_loss, 'total_actor_loss_with_reg')

        all_but_last_row = self.all_but_last_actor_layer

        total_loss = tf.check_numerics(total_loss, 'nan loss is being created')
        # total_loss = tf.Print(total_loss, [total_loss], 'total_loss')

        actor_gradients = self.optimizer.compute_gradients(total_loss,
                                                           all_but_last_row)

        merged_gradient_list += actor_gradients

        return merged_gradient_list, total_loss, actor_reg_loss

    def create_split_actor_loss(self, index, logprobs, taken_actions, advantages, actor_network_variables):
        if len(taken_actions.get_shape()) == 2:
            taken_actions = tf.squeeze(taken_actions, axis=[1])

        # calculates the entropy loss from getting the label wrong
        cross_entropy_loss, wrongNess, reduced = self.calculate_loss_of_actor(logprobs, taken_actions, index)
        if not reduced:
            tf.summary.histogram('actor_wrongness', wrongNess)
        with tf.name_scope("compute_pg_gradients"):
            pg_loss = cross_entropy_loss * (wrongNess * wrongNess)

            pg_loss = tf.check_numerics(pg_loss, 'nan pg_loss')

            if reduced:
                pg_loss = tf.reduce_mean(pg_loss, name='pg_loss')
                tf.summary.scalar("actor_x_entropy_loss", cross_entropy_loss)
            else:
                tf.summary.scalar("actor_x_entropy_loss", tf.reduce_mean(cross_entropy_loss))

            actor_reg_loss = self.get_regularization_loss(actor_network_variables, prefix="actor")

            actor_loss = pg_loss + actor_reg_loss * self.reg_param

            # compute actor gradients
            actor_gradients = self.optimizer.compute_gradients(actor_loss,
                                                               [actor_network_variables])

            # compute policy gradients
            for i, (grad, var) in enumerate(actor_gradients):
                if grad is not None:
                    actor_gradients[i] = (grad * advantages, var)

            if reduced:
                tf.summary.scalar("actor_loss", actor_loss)
            else:
                tf.summary.scalar("actor_loss", tf.reduce_mean(actor_loss))
            return [actor_gradients, actor_loss]

    def create_critic_gadients(self):
        critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.critic_network_variables],
                                        name='critic_reg_loss')

        tf.summary.scalar("critic_reg_loss", critic_reg_loss)

        # compute critic gradients
        mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values), name='mean_square_loss')

        critic_loss = mean_square_loss + self.reg_param * critic_reg_loss
        tf.summary.scalar("critic_loss", critic_loss)
        critic_gradients = self.optimizer.compute_gradients(critic_loss, self.critic_network_variables)
        return (critic_gradients, critic_loss, critic_reg_loss)

    def add_histograms(self, gradients):
        # summarize gradients
        for grad, var in gradients:
            tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + '/gradients', grad)

        # emit summaries
        tf.summary.histogram("estimated_values", self.estimated_values)

    def _compute_training_op(self, actor_gradients, critic_gradients):
        # collect all gradients
        gradients = actor_gradients + critic_gradients

        # clip gradients
        for i, (grad, var) in enumerate(gradients):
            # clip gradients by norm
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

        self.add_histograms(gradients)
        # training update
        with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
            return self.optimizer.apply_gradients(gradients)

    def create_reward(self):
        return None

    def discount_rewards(self, input_rewards, input):
        return self.reward_manager.create_reward_graph(input)

    #def parse_actions(self, taken_actions):
    #    return tf.cast(self.action_handler.create_indexes_graph(taken_actions), tf.int32)

    def run_train_step(self, calculate_summaries, input_states, actions, rewards=None):
        # perform one update of training
        if self.batch_size > self.mini_batch_size:
            self.sess.run([self.input, self.taken_actions, self.iterator.initializer],
                          feed_dict={self.input_placeholder: input_states, self.taken_actions_placeholder: actions})
            num_batches = math.ceil(float(self.batch_size) / float(self.mini_batch_size))
            # print('num batches', num_batches)
            counter = 0
            while counter < num_batches:
                try:
                    result, summary_str = self.sess.run([
                        self.train_op,
                        self.summarize if calculate_summaries else self.no_op
                    ])
                    # emit summaries
                    if calculate_summaries:
                        self.summary_writer.add_summary(summary_str, self.train_iteration)
                        self.train_iteration += 1
                    counter += 1
                except tf.errors.OutOfRangeError:
                    #print("End of training dataset.")
                    break
            print('batch amount:', counter)
        else:
            result, summary_str = self.sess.run([
                    self.train_op,
                    self.summarize if calculate_summaries else self.no_op
                ],
                feed_dict={
                    self.input_placeholder: input_states,
                    self.taken_actions_placeholder: actions
                })
            # emit summaries
            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.train_iteration,
                    )
                self.train_iteration += 1

        return None, None


    def get_model_name(self):
        return 'a_c_policy_gradient' + ('_split' if self.action_handler.is_split_mode else '') + str(self.num_layers) + '-layers'

    def calculate_loss_of_actor(self, logprobs, taken_actions, index):
        """
        Calculates the loss of th
        :param cross_entropy_loss:
        :return: The calculated_tensor, If the result is a scalar.
        """
        return self.action_handler.get_action_loss_from_logits(logprobs, taken_actions, index), 1.0, True

