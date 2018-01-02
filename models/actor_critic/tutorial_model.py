from models.actor_critic.policy_gradient import PolicyGradient
import tensorflow as tf


class TutorialModel(PolicyGradient):

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
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)

    def create_training_op(self, logprobs, labels):
        actor_gradients, actor_loss, actor_reg_loss = self.create_actor_gradients(logprobs, labels)

        tf.summary.scalar("total_reg_loss", actor_reg_loss)

        return self._compute_training_op(actor_gradients, [])

    def create_advantages(self):
        return tf.constant(1.0)