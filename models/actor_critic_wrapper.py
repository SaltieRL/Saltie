from models.actorcritic import PolicyGradientActorCritic
import tensorflow as tf

class ActorCriticModel:

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 is_training=False,
                 summary_writer=None,
                 summary_every=100,
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):

        self.session = session
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.pg_reinforce = PolicyGradientActorCritic(session,
                                                      optimizer,
                                                      self.actor_network,
                                                      self.critic_network,
                                                      self.state_dim,
                                                      self.num_actions,
                                                      summary_writer=summary_writer)
    def initialize_model(self):
        pass

    def store_rollout(self, state, last_action, reward):
        self.pg_reinforce.store_rollout(state, last_action, reward)


    def sample_action(self, states):
        return self.pg_reinforce.sampleAction(states)
