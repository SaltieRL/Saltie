import tensorflow as tf


class BaseModelCreator:
    def get_input_layer(self, input_feed, model_parameters):
        """
        Creates the tensorflow input layer
        :param input_feed: The input
        :return: A tensor that is the first layer of the model
        """
        raise NotImplementedError("Need to implement get_input_layer")

    def get_hidden_layer(self, input_layer, model_parameters):
        """
        :param input_layer: The input layer to the model
        :param model_parameters: Not sure yet (probably a v4 config object)
        :return: A tensor that represents the last hidden layer(s) of the model
        """
        raise NotImplementedError("Need to implement get_input_layer")

    def get_output_layer(self, hidden_layer, output_dim, model_parameters):
        """
        Gets the output layer for the model.
        :param hidden_layer: The hidden layers
        :param output_dim: The dimension of this output layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :return: The last layer of the model and the prediction layer of the model
        """
        last_layer = self.get_last_layer(hidden_layer, output_dim, model_parameters)
        return last_layer, self.get_prediction_layer(last_layer, output_dim, model_parameters)

    def get_last_layer(self, hidden_layer, output_dim, model_parameters):
        """
        Gets the last layer for the model.
        :param hidden_layer: The hidden layers
        :param output_dim: The dimension of this layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :return: The last layer of the model.
        """
        raise NotImplementedError("Need to implement get_input_layer")

    def get_prediction_layer(self, last_layer, output_dim, model_parameters):
        """
        Gets the output layer for the model.
        :param last_layer: The hidden layers
        :param output_dim: The dimension of this output layer
        :param model_parameters: Not sure yet (probably a v4 config object)
        :return: The prediction layer of the model.
        """
        raise NotImplementedError("Need to implement get_input_layer")
