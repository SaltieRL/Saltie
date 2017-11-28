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

    def create_training_op(self):
        with tf.name_scope("compute_pg_gradients"):
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.actor_network_variables])
            self.actor_loss = self.pg_loss + self.reg_param * self.actor_reg_loss

            # compute actor gradients
            self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, self.actor_network_variables)
            # compute advantages A(s) = R - V(s)
            self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
            # compute policy gradients
            for i, (grad, var) in enumerate(self.actor_gradients):
                if grad is not None:
                    self.actor_gradients[i] = (grad * self.advantages, var)

            # compute critic gradients
            self.mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values))
            self.critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.critic_network_variables])
            self.critic_loss = self.mean_square_loss + self.reg_param * self.critic_reg_loss
            self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, self.critic_network_variables)

            # collect all gradients
            self.gradients = self.actor_gradients + self.critic_gradients

            # clip gradients
            for i, (grad, var) in enumerate(self.gradients):
                # clip gradients by norm
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            # summarize gradients
            for grad, var in self.gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)

            # emit summaries
            tf.summary.histogram("estimated_values", self.estimated_values)
            tf.summary.scalar("actor_loss", self.actor_loss)
            tf.summary.scalar("critic_loss", self.critic_loss)
            tf.summary.scalar("reg_loss", self.actor_reg_loss + self.critic_reg_loss)

        # training update
        with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
            self.train_op = self.optimizer.apply_gradients(self.gradients)
