from conversions import input_formatter
import tensorflow as tf


class TensorflowInputFormatter(input_formatter.InputFormatter):
    def __init__(self, team, index, batch_size):
        super().__init__(team, index)
        self.batch_size = batch_size

    def get_last_touched_ball(self, car, latest_touch):
        return tf.equal(car.wName, latest_touch.wPlayerName)

    def return_emtpy_player_array(self):
        """
        :return: An array representing a car with no data
        """
        array = []
        for i in range(len(super().return_emtpy_player_array())):
            array.append(tf.constant([0.0] * self.batch_size))
        return array

    def create_result_array(self, array):
        converted_array = []
        for i in range(len(array)):
            casted_number = tf.cast(array[i], tf.float32)
            #casted_number = tf.Print(casted_number, [tf.shape(casted_number)], 'index: ' + str(i) + ' name: ' + array[i].name + ' ')
            converted_array.append(casted_number)
        result = tf.stack(converted_array)
        return tf.reshape(result, [self.batch_size, tf.shape(result)[0]])
