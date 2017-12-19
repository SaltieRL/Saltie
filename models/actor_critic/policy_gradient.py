import tensorflow as tf
from models.actor_critic.base_actor_critic import BaseActorCritic
from modelHelpers import tensorflow_reward_manager


class PolicyGradient(BaseActorCritic):

    reg_param = 0.001
    max_gradient = 1

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        self.reward_manager = tensorflow_reward_manager.TensorflowRewardManager()

        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)


    def create_training_op(self, logprobs, taken_actions):
        critic_gradients, critic_loss, critic_reg_loss = self.create_critic_gadients()
        actor_gradients, actor_loss, actor_reg_loss = self.create_actor_gradients(logprobs, taken_actions)

        tf.summary.scalar("total_reg_loss", critic_reg_loss + actor_reg_loss)

        return self._compute_training_op(actor_gradients, critic_gradients)

    def create_actor_gradients(self, logprobs, taken_actions):
        advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values, name='advantages')

        all_actor_variables = self.actor_network_variables + self.actor_last_row_layer

        actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in all_actor_variables],
                                       name='actor_reg_loss')

        tf.summary.scalar("actor_reg_loss", actor_reg_loss)


        result = self.action_handler.run_func_on_split_tensors([logprobs,
                                                                taken_actions,
                                                                actor_reg_loss,
                                                                advantages,
                                                                self.last_row_variables],
                                                               self.create_split_actor_loss,
                                                               return_as_list=True)

        merged_gradient_list = []
        merged_loss_list = []
        for item in result:
            merged_gradient_list += item[0]
            merged_loss_list.append(item[1])

        total_loss = tf.reduce_sum(tf.stack(merged_loss_list))

        all_but_last_row = self.actor_network_variables

        actor_gradients = self.optimizer.compute_gradients(total_loss,
                                                           all_but_last_row)

        merged_gradient_list += actor_gradients

        tf.summary.scalar("toal_actor_loss", total_loss)

        return merged_gradient_list, total_loss, actor_reg_loss


    def create_split_actor_loss(self, logprobs, taken_actions, actor_reg_loss, advantages, actor_network_variables):
        if len(taken_actions.get_shape()) == 2:
            taken_actions = tf.squeeze(taken_actions)

        # calculates the entropy loss from getting the label wrong
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs,
                                                                            labels=taken_actions)
        with tf.name_scope("compute_pg_gradients"):
            pg_loss = tf.reduce_mean(cross_entropy_loss, name='pg_loss')

            actor_loss = pg_loss + self.reg_param * actor_reg_loss

            # compute actor gradients
            actor_gradients = self.optimizer.compute_gradients(actor_loss,
                                                               [actor_network_variables])
            # compute advantages A(s) = R - V(s)

            # compute policy gradients
            for i, (grad, var) in enumerate(actor_gradients):
                if grad is not None:
                    actor_gradients[i] = (grad * advantages, var)

            tf.summary.scalar("actor_loss", actor_loss)
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

    def _compute_training_op(self, actor_gradients, critic_gradients):

        # collect all gradients
        gradients = actor_gradients + critic_gradients

        # clip gradients
        for i, (grad, var) in enumerate(gradients):
            # clip gradients by norm
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

        # summarize gradients
        for grad, var in gradients:
            tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + '/gradients', grad)

        # emit summaries
        tf.summary.histogram("estimated_values", self.estimated_values)

            # training update
        with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
            return self.optimizer.apply_gradients(gradients)

    def create_reward(self):
        return None

    def discount_rewards(self, input_rewards):
        return self.reward_manager.create_reward_graph(self.input)

    #def parse_actions(self, taken_actions):
    #    return tf.cast(self.action_handler.create_indexes_graph(taken_actions), tf.int32)

    def run_train_step(self, calculate_summaries, input_states, actions, rewards):
        # perform one update of training

        feed = {
            self.input: input_states,
            self.taken_actions: actions
        }

        result, summary_str = self.sess.run([
            self.train_op,
            self.summarize if calculate_summaries else self.no_op
        ], feed_dict=feed)
        return result, summary_str

    def get_model_name(self):
        return 'a_c_policy_gradient' + ('_split' if self.action_handler.is_split_mode else '') + str(self.num_layers) + '-layers'
