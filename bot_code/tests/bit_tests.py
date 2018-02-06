import tensorflow as tf
import numpy as np

def fancy_calculate_number_of_ones(number):
    """Only use this once it is supported"""

    # https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
    #i = i - ((i >> 1) & 0x55555555);
    #i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    #return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

    # (i + (i >> 4))

    fivers = tf.constant(0x55555555, dtype=tf.int32)
    threes = tf.constant(0x33333333, dtype=tf.int32)
    ffs = tf.constant(0x0F0F0F0F, dtype=tf.int32)
    ones = tf.constant(0x01010101, dtype=tf.int32)
    threes_64 = tf.constant(0o033333333333, dtype=tf.int64)
    full_ones = tf.constant(0o011111111111, dtype=tf.int64)
    sevens = tf.constant(0o030707070707, dtype=tf.int64)

    #i = i - ((i >> 1) & 0x55555555);
    i = number - tf.bitwise.bitwise_and(tf.bitwise.right_shift(number, 1), fivers)

    #i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = tf.bitwise.bitwise_and(tf.bitwise.right_shift(i, 1), threes) + \
        tf.bitwise.bitwise_and(tf.bitwise.right_shift(i, 2), threes)

    # (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101)
    i = tf.bitwise.bitwise_and(i + tf.bitwise.right_shift(i, 4), ffs) * ones

    # i >> 24
    i = tf.bitwise.right_shift(i, 24)


    number = tf.cast(number, tf.int64)
    bitwise1 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(number, 1), threes_64)
    bitwise2 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(number, 2), full_ones)
    uCount = number - bitwise1 - bitwise2

    bitwise3 = tf.bitwise.bitwise_and(uCount + tf.bitwise.right_shift(uCount, 3), sevens)
    result = tf.mod(bitwise3, 63)
    #result = tf.Print(result, [result, i])
    i = tf.Print(i, [number, i, result])
    #return result
    return i



def test1():

    session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))

    input = tf.placeholder(tf.int32, shape=[1])

    result = fancy_calculate_number_of_ones(input)
    for i in range(17):
        array = np.array([i])
        session.run(result, feed_dict={input: array})



def test2():

    session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))

    input = tf.placeholder(tf.int32, shape=[1])

    result = tf.bitwise.right_shift(input, 2)
    divide = input // 4

    result = tf.Print(result, [input, result, divide])
    for i in range(17):
        array = np.array([i])
        session.run(result, feed_dict={input: array})




if __name__ == '__main__':
    #test1()
    test2()
