import struct
import io
import numpy as np


def convert_numpy_array(numpy_array):
    """
    Converts a numpy array into compressed bytes
    :param numpy_array: An array that is going to be converted into bytes
    :return: A BytesIO object that contains compressed bytes
    """
    compressed_array = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
    np.savez_compressed(compressed_array, numpy_array)
    return compressed_array


def read_data(file, process_pair_function):
    """
    Reads a file.  Quits if anything breaks.
    :param file: A simple python file object that will be read
    :param process_pair_function: A function that takes in an input array and an output array.
    There is also an optional number saying how many times this has been called for a single file.
    It always starts at 0
    :return: None
    """

    pair_number = 0
    while True:
        try:
            chunk = file.read(4)
            if chunk == '':
                break
            input_array = get_array(file, chunk)
            chunk = file.read(4)
            if chunk == '':
                break
            output_array = get_array(file, chunk)
            process_pair_function(input_array, output_array, pair_number)
            pair_number += 1
        except EOFError:
            print('reached end of file')
            break


def get_array(file, chunk):
    """
    Gets a compressed numpy array from a file.

    Throws an EOFError if it has problems loading the data.

    :param file: The file that is being read
    :param chunk: A chunk representing a single number, this will be the number of bytes the array takes up.
    :return: A numpy array
    """
    try:
        starting_byte = struct.unpack('i', chunk)[0]
    except struct.error:
        print('struct error')
        raise EOFError
    numpy_bytes = file.read(starting_byte)
    fake_file = io.BytesIO(numpy_bytes)
    try:
        result = np.load(fake_file)
    except OSError:
        print('numpy parse error')
        raise EOFError
    return result[result.files[0]]


def default_process_pair(input_array, output_array, pair_number):
    pass


if __name__ == '__main__':
    test_file = open('training\\1511676262748-30\\SaltieRl(2).txt', 'r+b')
    print(test_file)
    read_data(test_file, default_process_pair)
