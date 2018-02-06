import numpy as np
import tensorflow as tf

from bot_code.models import base_model
from bot_code.modelHelpers import tensorflow_reward_manager
from bot_code.models import SplitLayers


class PolicyGradient(SplitLayers):
    max_gradient = 1.0
    total_loss_divider = 1.0

    def __init__(self, session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None
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
        self.reward_manager = tensorflow_reward_manager.TensorflowRewardManager(self.state_dim)

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

        total_loss += actor_reg_loss

        # total_loss = tf.Print(total_loss, [total_loss], 'total_loss')

        total_loss = tf.identity(total_loss, 'total_actor_loss_with_reg')

        all_but_last_row = self.all_but_last_actor_layer

        # total_loss = tf.Print(total_loss, [total_loss], 'total_loss')

        actor_gradients = self.optimizer.compute_gradients(total_loss,
                                                           all_but_last_row)

        merged_gradient_list += actor_gradients

        return merged_gradient_list, total_loss, actor_reg_loss

    def create_split_actor_loss(self, index, logprobs, taken_actions, advantages, actor_network_variables):
        if len(taken_actions.get_shape()) == 2:
            taken_actions = tf.squeeze(taken_actions, axis=[1])

        # calculates the entropy loss from getting the label wrong
        cross_entropy_loss, wrongness, reduced = self.calculate_loss_of_actor(logprobs, taken_actions, index)
        if reduced:
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
        if not reduced:
            if self.action_handler.is_classification(index):
                tf.summary.histogram('actor_wrongness', wrongness)
            else:
                tf.summary.histogram('actor_wrongness', cross_entropy_loss)
        with tf.name_scope("compute_pg_gradients"):
            actor_loss = cross_entropy_loss * (wrongness * wrongness)

            # actor_loss = tf.check_numerics(actor_loss, 'nan pg_loss')

            if reduced:
                actor_loss = tf.reduce_mean(actor_loss, name='pg_loss')
                tf.summary.scalar(self.action_handler.get_loss_type(index), cross_entropy_loss)
            else:
                tf.summary.scalar(self.action_handler.get_loss_type(index), tf.reduce_mean(cross_entropy_loss))

            actor_reg_loss = self.get_regularization_loss(actor_network_variables, prefix="actor")

            actor_loss = actor_loss + actor_reg_loss

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
        critic_reg_loss = self.get_regularization_loss(self.critic_network_variables, prefix='critic')
        # compute critic gradients
        mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values), name='mean_square_loss')

        critic_loss = mean_square_loss + critic_reg_loss
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
                post_nanning = tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
                gradients[i] = (post_nanning, var)

        # graph before we clip gradients
        self.add_histograms(gradients)

        for i, (grad, var) in enumerate(gradients):
            # clip gradients by norm
            if grad is not None:
                post_clipping = tf.clip_by_norm(grad, self.max_gradient)
                gradients[i] = (post_clipping, var)

        # training update
        with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
            return self.optimizer.apply_gradients(gradients)

    def create_reward(self):
        return None

    def discount_rewards(self, input_rewards, input):
        return self.reward_manager.create_reward_graph(input)

    def get_model_name(self):
        return 'a_c_policy_gradient' + ('_split' if self.action_handler.is_split_mode else '') + str(self.num_layers) + '-layers'

    def calculate_loss_of_actor(self, logprobs, taken_actions, index):
        """
        Calculates the loss of th
        :param cross_entropy_loss:
        :return: The calculated_tensor, If the result is a scalar.
        """
        return self.action_handler.get_action_loss_from_logits(logprobs, taken_actions, index), 1.0, True

