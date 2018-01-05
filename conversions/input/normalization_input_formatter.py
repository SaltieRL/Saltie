import tensorflow as tf
from conversions.input import tensorflow_input_formatter


class NormalizationInputFormatter(tensorflow_input_formatter.TensorflowInputFormatter):
    def __init__(self, team, index, batch_size):
        super().__init__(team, index, batch_size)

    def create_result_array(self, array):
        converted_array = []
        for i in range(len(array)):
            casted_number = tf.cast(array[i], tf.float32)
            converted_array.append(casted_number)
        result = tf.stack(converted_array)
        return result

    def get_score_info(self, score, diff_in_score):
        result = super().get_score_info(score, diff_in_score)

        # the change in score can only be -1 to 1
        result[len(result) - 1] = [-1, 1]
        return result
