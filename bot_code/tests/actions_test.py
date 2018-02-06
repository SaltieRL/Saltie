from bot_code.modelHelpers.actions import action_factory
import tensorflow as tf
import numpy as np

def test1():
    """
    Test that handler and split handler return the same results
    :return:
    """
    handler = action_factory.get_handler(False)
    split_handler = action_factory.get_handler(True, action_factory.default_scheme)

    session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))

    input = np.array([[ 1.0, -1.0,  0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [-1.0,  1.0, -0.5,  0.5, 0.0, 0.0, 0.0, 1.0],
                      [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                      [-0.8, -0.4,  0.1,  0.6, 0.0, 0.0, 0.0, 0.0],
                      [-0.2, -0.3,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 0.0, 0.0]])

    #t, y, p, r,
    real_action = tf.Variable(input, dtype=tf.float32)

    result = handler.create_action_indexes_graph(real_action)
    result_split = handler.create_action_indexes_graph(real_action)

    init = tf.global_variables_initializer()
    session.run(init)

    indexes, indexes_split = session.run(result, result_split)

    for index in range(5):
        row = input[index]
        print('blank row')
        # print('input row    ', np.array(row, dtype=np.float32))
        result = handler.create_action_index(row)
        split_result = split_handler.create_action_index(row)
        print('numpy result ', np.array(result, dtype=np.float32))
        print('tensor result', np.array(indexes[index], dtype=np.float32))
        print('split numpy result ', np.array(split_result, dtype=np.float32))
        print('split tensor result', np.array(indexes_split[index], dtype=np.float32))


def test2():
    """
    Test that handler and dynamic handler return the same results
    :return:
    """
    handler = action_factory.get_handler(False)
    dynamic_handler = action_factory.get_handler(True, dynamic_action_handler.super_split_scheme)
    # dynamic_handler2 = dynamic_action_handler.DynamicActionHandler(dynamic_action_handler.current_scheme)

    session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))

    input = np.array([[ 1.0, -1.0,  0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [-1.0,  1.0, -0.5,  0.5, 0.0, 0.0, 0.0, 1.0],
                      [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                      [-0.8, -0.4,  0.1,  0.6, 0.0, 0.0, 0.0, 0.0],
                      [-0.2, -0.3,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 0.0, 0.0]])

    #t, y, p, r,
    real_action = tf.Variable(input, dtype=tf.float32)

    result = handler.create_action_indexes_graph(real_action)
    result2 = dynamic_handler.create_action_indexes_graph(real_action)

    init = tf.global_variables_initializer()
    session.run(init)

    indexes, dynamic_indexes = session.run([result, result2])

    for index in range(5):
        row = input[index]
        print('blank row')
        #     print('input row    ', np.array(row, dtype=np.float32))
        result = handler.create_action_index(row)
        dynamic_result = dynamic_handler.create_action_index(row)
        print('numpy result ', np.array(result, dtype=np.float32))
        print('tensor result', np.array(indexes[index], dtype=np.float32))
        print('dynamic numpy result', np.array(dynamic_result, dtype=np.float32))
        print('dynamic tensor result', np.array(dynamic_indexes[index], dtype=np.float32))

        print('and back again')
        print('correct answer', row)
        print('numpy result', handler.create_controller_from_selection(result))
        # purposely using the working result
        print('dynamic result', dynamic_handler.create_controller_from_selection(dynamic_result))


def test3():
    handler = action_factory.get_handler(False)
    dynamic_handler = action_factory.get_handler(True, action_factory.regression_controls)

    session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))

    input = np.array([[ 1.0, -1.0,  0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [-1.0,  1.0, -0.5,  0.5, 0.0, 0.0, 0.0, 1.0],
                      [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                      [-0.8, -0.4,  0.1,  0.6, 0.0, 0.0, 0.0, 0.0],
                      [-0.2, -0.3,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0],
                      [-0.2, -0.3,  0.2,  0.3, 0.0, 1.0, 0.0, 0.0],
                      [ 1.0, -0.3,  0.2,  0.3, 0.0, 0.0, 1.0, 0.0],
                      [-1.0, -0.3,  0.2,  0.3, 0.0, 0.0, 0.0, 1.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 0.0, 0.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 1.0, 1.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 1.0, 0.0, 1.0],
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 1.0, 1.0, 0.0]])

    #t, y, p, r,
    real_action = tf.Variable(input, dtype=tf.float32)

    action_index = dynamic_handler.create_action_indexes_graph(real_action)
    back_again = dynamic_handler.create_tensorflow_controller_from_selection(tf.transpose(action_index), batch_size=len(input))

    init = tf.global_variables_initializer()
    session.run(init)

    indexes, dynamic_results = session.run([action_index, back_again])

    for index in range(len(input)):
        row = input[index]
        print('blank row')
        #     print('input row    ', np.array(row, dtype=np.float32))
        action_index = handler.create_action_index(row)
        print('numpy result ', np.array(action_index, dtype=np.float32))
        print('dynamic result', np.array(indexes[index], dtype=np.float32))

        print('and back again')
        print('correct answer', row)
        print('numpy result', dynamic_handler.create_controller_from_selection(indexes[index]))
        # purposely using the working result
        print('dynamic result', dynamic_results[index])

if __name__ == '__main__':
   # test1()
   # test2()
   test3()
