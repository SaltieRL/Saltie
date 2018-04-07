from bot_code.model_creator.base_model_creator import BaseModelCreator


class BaseAgentModelCreator(BaseModelCreator):

    def get_split_hidden_layer(self, hidden_layer, model_parameters, action_handler, action_index):
        """
        Gets the layers that have been split for each individual action
        :param hidden_layer: The non split layers
        :param model_parameters: Not sure yet (probably a v4 config object)
        :param action_handler: The action handler used with the action index to get action types.
        :param action_index: This is the index related to the specific action of this model.
        :return: A split layer
        """

    def get_output_layer(self, hidden_layer, output_dim, model_parameters, action_handler=None, action_index=-1):
        """
        Gets the output layer for the model.
        :param hidden_layer: The hidden layers
        :param output_dim: The dimension of this output layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :param action_handler: The action handler used with the action index to get action types.
        :param action_index: This is the index related to the specific action of this model.
        :return: The last layer of the model and the prediction layer of the model
        """
        last_layer = self.get_last_layer(hidden_layer, output_dim, model_parameters,
                                         action_handler=action_handler, action_index=action_index)
        return last_layer, self.get_prediction_layer(last_layer, output_dim, model_parameters,
                                                     action_handler=action_handler, action_index=action_index)

    def get_last_layer(self, hidden_layer, output_dim, model_parameters, action_handler=None, action_index=-1):
        """
        Gets the last layer for the model.
        :param hidden_layer: The hidden layers
        :param output_dim: The dimension of this layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :param action_handler: The action handler used with the action index to get action types.
        :param action_index: This is the index related to the specific action of this model.
        :return: The last layer of the model.
        """
        raise NotImplementedError("Need to implement get_input_layer")

    def get_prediction_layer(self, last_layer, output_dim, model_parameters, action_handler=None, action_index=-1):
        """
        Gets the output layer for the model.
        :param last_layer: The hidden layers
        :param output_dim: The dimension of this output layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :param action_handler: The action handler used with the action index to get action types.
        :param action_index: This is the index related to the specific action of this model.
        :return: The prediction layer of the model.
        """
        raise NotImplementedError("Need to implement get_input_layer")
