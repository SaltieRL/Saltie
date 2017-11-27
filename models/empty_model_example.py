class EmptyModelExample:
    """"
    This will be the example model framework with the needed functions but none of the code inside them
    You can copy this to implement your own model
    """
    def __init__(self, session,
                 state_dim,
                 num_actions,
                 is_training=False,
                 summary_writer=None,
                 summary_every=100):
        print('i do nothing!')

    def store_rollout(self, state, last_action, reward):
        print(' i do nothing!')

    def sample_action(self, states):
        #always return an integer
        return 10

    def create_training_model_copy(self, batch_size):
        #return a loss function in tensorflow
        loss = 0
        #return a placeholder for input data
        input = 0
        #return a placeholder for labeled data
        labels = 0
        return loss, input, labels
