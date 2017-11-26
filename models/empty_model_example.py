class EmptyModelExample:
    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session,
                 num_actions,
                 state_dim,
                 summary_writer=None,
                 summary_every=100):
        print('i do nothing!')

    def store_rollout(self, state, last_action, reward):
        print(' i do nothing!')

    def sample_action(self, states):
        print(' i do nothing!')
