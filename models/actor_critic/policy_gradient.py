import tensorflow as tf
from models.actor_critic.base_actor_critic import BaseActorCritic
from modelHelpers import tensorflow_reward_manager


class PolicyGradient(BaseActorCritic):

    reg_param = 0.001
    max_gradient = 5

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        self.reward_manager = tensorflow_reward_manager.TensorflowRewardManager()

        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)


    def create_training_op(self, cross_entropy_loss, estimated_values, discounted_rewards, actor_network_variables, critic_network_variables):
        with tf.name_scope("compute_pg_gradients"):
            pg_loss = tf.reduce_mean(cross_entropy_loss, name='pg_loss')
            actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables],
                                           name='actor_reg_loss')
            actor_loss = pg_loss + self.reg_param * actor_reg_loss

            # compute actor gradients
            actor_gradients = self.optimizer.compute_gradients(actor_loss, actor_network_variables)
            # compute advantages A(s) = R - V(s)
            advantages = tf.reduce_sum(discounted_rewards - estimated_values, name='advantages')
            # compute policy gradients
            for i, (grad, var) in enumerate(actor_gradients):
                if grad is not None:
                    actor_gradients[i] = (grad * advantages, var)

            # compute critic gradients
            mean_square_loss = tf.reduce_mean(tf.square(discounted_rewards - estimated_values), name='mean_square_loss')
            critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables],
                                            name='critic_reg_loss')
            critic_loss = mean_square_loss + self.reg_param * critic_reg_loss
            critic_gradients = self.optimizer.compute_gradients(critic_loss, critic_network_variables)

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
            tf.summary.scalar("actor_loss", actor_loss)
            tf.summary.scalar("critic_loss", critic_loss)
            tf.summary.scalar("reg_loss", actor_reg_loss + critic_reg_loss)

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

    def run_train_step(self, calculate_summaries, input_states, actions, rewards):
        # perform one update of training
        self.sess.run([self.input, self.taken_actions, self.iterator.initializer],
                      feed_dict={self.input_placeholder:input_states, self.taken_actions_placeholder: actions})

        while True:
            try:
                result, summary_str = self.sess.run([
                    self.train_op,
                    self.summarize if calculate_summaries else self.no_op
                ])
                # emit summaries
                if calculate_summaries:
                    self.summary_writer.add_summary(summary_str, self.train_iteration)
            except tf.errors.OutOfRangeError:
                #print("End of training dataset.")
                break
        return None, None


    def get_model_name(self):
        return 'a_c_policy_gradient' + ('_split' if self.action_handler.is_split_mode else '') + str(self.num_layers) + '-layers'
