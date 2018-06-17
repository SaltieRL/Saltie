class BaseModel:
    """The base model, this will internally hold different tensorflow/keras models"""

    def get_input_layer(self):
        """Creates the input layer of the model, takes in feeding dicts"""
        pass

    def create_hidden_layers(self):
        """Creates the internal hidden layers if needed"""
        pass

    def create_output_layer(self):
        """Creates the output layer of the model.
        :return The output layer of the model"""
