import numpy as np
import tensorflow as tf

from bot_code.modelHelpers import tensorflow_feature_creator
from bot_code.models.base_model import BaseModel
from bot_code.conversions.input.input_formatter import InputFormatter


class BaseAgentModel(BaseModel):
    """
    A base class for any model that outputs car actions given the the current state (+features).
    """
    def __init__(self,
                 session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None):
        # TODO: Should index here always == player_index?
        (team, index) = input_formatter_info[0], input_formatter_info[1]

        self.input_formatter = InputFormatter(team, index)
        self.state_dim = self.input_formatter.get_state_dim()
        self.state_feature_dim = self.state_dim
        self.num_actions = num_actions

        # player index used for live graphing
        self.player_index = player_index

        # for interfacing with the rest of the world
        self.action_handler = action_handler

        super().__init__(session,
                         input_dim=self.state_dim,
                         output_dim=self.num_actions,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

    def create_input_array(self, game_tick_packet, frame_time):
        """Creates the input array from the game_tick_packet"""
        return self.input_formatter.create_input_array(game_tick_packet, frame_time)

    def apply_feature_creation(self, feature_creator):
        self.state_feature_dim = self.state_dim + tensorflow_feature_creator.get_feature_dim()
        self.feature_creator = feature_creator

    def sample_action(self, input_state):
        """
        Runs the model to get a single action that can be returned.
        :param input_state: This is the current state of the model at this point in time.
        :return:
        A sample action that can then be used to get controller output.
        """
        return self.sess.run(self.get_agent_output(), feed_dict=self.create_sampling_feed_dict(input_state))

    def create_sampling_feed_dict(self, input_array):
        """
        :param input_array: The array that goes into the input.
        :return:
        """
        return {self.get_input_placeholder(): input_array,
                self.batch_size_placeholder: [len(input_array)]}

    def get_agent_output(self):
        """
        :return: A tensor representing the output of the agent
        """
        return self.model

    def split_logits(self, logits):
        """
        :param logits: The logits that are passed into #_create_training_op (returned from #_create_model)
        :return: A version of the logits that can be passed into #_create_split_training_op
        """
        return logits

    def get_split_training_parameters(self):
        """
        Any extra parameters that are wanting to be passed into #_create_split_training_op
        :return: A list of any other parameters that should be passed.
        """
        return []

    def _create_training_op(self, predictions, logits, raw_model_input, labels):
        split_logits = self.split_logits(logits)

        indexes = np.arange(0, len(self.action_handler.get_action_sizes()), 1).tolist()

        parameters = [indexes,
                      split_logits,
                      labels]
        parameters += self.get_split_training_parameters()

        central_result = self._create_central_training_op(predictions, logits, raw_model_input, labels)

        split_result = self.action_handler.run_func_on_split_tensors(parameters,
                                                               self._create_split_training_op,
                                                               return_as_list=True)

        return self._process_results(central_result, split_result)

    def _create_central_training_op(self, predictions, logits, raw_model_input, labels):
        """
        Called to create a specific training operation for this one model.
        This should be overwritten by subclasses.
        :param predictions: This is the part of the model that can be used externally to produce predictions
        :param logits: The last layer of the model itself, this is typically the layer before argmax is applied.
        :param raw_model_input: This is an unmodified input that can be used for training uses. (it is batched)
        :param labels: These are the labels that can be used to generate loss
        :return: Something that will be passed into #_process_results
        """
        raise NotImplementedError('Derived classes must override this.')

    def _create_split_training_op(self, indexes, logits, labels, *args):
        """
        Called for each individual action.
        :param indexes: This is the action index in the list of actions, can be used with the action handler.
        :param logits: A split version of the logits, this only applies to one particular action set
        :param labels: A split version of the labels this only applies to one particular action set
        :param args: A list of custom actions
        :return: Something that will be passed into #_process_results as part of a list
        """
        raise NotImplementedError('Derived classes must override this.')

    def _process_results(self, central_result, split_result):
        """
        Processes the results of the central and split training operations
        :param central_result: This is the central result it is a single item.
        :param split_result: This is a list of items with length equal to the number of actions
        :return: A tensorflow operation that is used in the training step
        """
        raise NotImplementedError('Derived classes must override this.')
