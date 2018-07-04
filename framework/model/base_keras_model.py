import tensorflow as tf
from framework.model.base_model import BaseModel


class BaseKerasModel(BaseModel):

    model = None

    def __init__(self, use_default_dense=True, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)):
        if use_default_dense:
            self.activation = activation
            self.kernel_regularizer = kernel_regularizer

    def create_input_layer(self, input_placeholder):
        """Creates keras model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_tensor=input_placeholder))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation))
        model.add(tf.keras.layers.Dense(32))

    def create_output_layer(self):
        # sigmoid/tanh all you want on self.model
        return self.model.layers[-1].output

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=1000)

    def predict(self, arr):
        return self.model.predict(arr)
