from modelHelpers import action_handler
import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    handler = action_handler.ActionHandler(split_mode=True)

    session = tf.Session()

    input = np.array([[ 1.0, -1.0,  0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [-1.0,  1.0, -0.5,  0.5, 0.0, 0.0, 0.0, 1.0],
                      [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                      [-0.8, -0.4,  0.1,  0.6, 0.0, 0.0, 0.0, 0.0],
                      [-0.2, -0.3,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 0.0, 0.0]])

    #t, y, p, r,
    real_action = tf.Variable(input, dtype=tf.float32)

    result = handler._create_split_indexes_graph(real_action)

    init = tf.global_variables_initializer()
    session.run(init)

    indexes = session.run(result)

    for index in range(5):
        row = input[index]
        print('blank row')
   #     print('input row    ', np.array(row, dtype=np.float32))
        result = handler.create_action_index(row)
        print('numpy result ', np.array(result, dtype=np.float32))
        print('tensor result', np.array(indexes[index], dtype=np.float32))

