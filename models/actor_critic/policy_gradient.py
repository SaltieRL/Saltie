import tensorflow as tf
from models.actor_critic.base_actor_critic import BaseActorCritic


class PolicyGradient(BaseActorCritic):

    reg_param = 0.001
    max_gradient = 5

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

    def create_training_op(self, cross_entropy_loss, estimated_values, actor_network_variables, critic_network_variables):
        with tf.name_scope("compute_pg_gradients"):
            pg_loss = tf.reduce_mean(cross_entropy_loss)
            actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables])
            actor_loss = pg_loss + self.reg_param * actor_reg_loss

            # compute actor gradients
            actor_gradients = self.optimizer.compute_gradients(actor_loss, actor_network_variables)
            # compute advantages A(s) = R - V(s)
            advantages = tf.reduce_sum(self.discounted_rewards - estimated_values)
            # compute policy gradients
            for i, (grad, var) in enumerate(actor_gradients):
                if grad is not None:
                    actor_gradients[i] = (grad * advantages, var)

            # compute critic gradients
            mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - estimated_values))
            critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
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
