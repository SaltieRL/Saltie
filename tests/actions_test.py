from modelHelpers.actions import action_handler, dynamic_action_handler
import tensorflow as tf
import numpy as np

def test1():
    handler = action_handler.ActionHandler(split_mode=True)

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



def test2():
    handler = action_handler.ActionHandler(split_mode=True)
    dynamic_handler = dynamic_action_handler.DynamicActionHandler(dynamic_action_handler.current_scheme)
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

    result = handler.create_indexes_graph(real_action)
    result2 = dynamic_handler.create_indexes_graph(real_action)

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
        print('numpy result', handler.create_controller_output_from_actions(result))
        # purposely using the working result
        print('dynamic result', dynamic_handler.create_controller_output_from_actions(result))


def test3():
    handler = action_handler.ActionHandler(split_mode=True)
    dynamic_handler = dynamic_action_handler.DynamicActionHandler(dynamic_action_handler.current_scheme)

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
                      [-0.25, -0.75,  0.25,  0.75, 0.0, 0.0, 0.0, 0.0]])

    #t, y, p, r,
    real_action = tf.Variable(input, dtype=tf.float32)

    result = handler._create_split_indexes_graph(real_action)
    back_again = dynamic_handler.create_tensorflow_controller_output_from_actions(tf.transpose(result), batch_size=9)

    init = tf.global_variables_initializer()
    session.run(init)

    indexes, dynamic_results = session.run([result, back_again])

    for index in range(9):
        row = input[index]
        print('blank row')
        #     print('input row    ', np.array(row, dtype=np.float32))
        result = handler.create_action_index(row)
        print('numpy result ', np.array(result, dtype=np.float32))
        print('tensor result', np.array(indexes[index], dtype=np.float32))

        print('and back again')
        print('numpy result', handler.create_controller_output_from_actions(result))
        # purposely using the working result
        print('dynamic result', dynamic_results[index])


if __name__ == '__main__':
   # test1()
   test2()
   # test3()
