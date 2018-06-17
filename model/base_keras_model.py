from model.base_model import BaseModel
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1


class BaseKerasModel(BaseModel):
    def __init__(self, use_default_dense=True, activation='relu', kernel_regularizer=l1(0.01)):
        if use_default_dense:
            self.activation = activation
            self.kernel_regularizer = kernel_regularizer

    def get_input_layer(self, input_placeholder):
        """Creates keras model"""
        kernel_regularizer = l1(0.01)
        model = Sequential()
        model.add(InputLayer(input_tensor=input_placeholder))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        model.add(Dropout(0.3))
        model.add(Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation)
        model.add(Dense(32))

    def create_output_layer(self):
        # sigmoid/tanh all you want on self.model
        return model.layers[-1].output
