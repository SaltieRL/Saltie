import struct
import io
import numpy as np
import os


EMPTY_FILE = 'empty'
NON_FLIPPED_FILE_VERSION = 0
FLIPPED_FILE_VERSION = 1

#BYTES_OBJECT = io.BytesIO()


def write_array_to_file(game_file, array):
    bytes = convert_numpy_array(array)
    size_of_bytes = len(bytes.getvalue())
    game_file.write(struct.pack('i', size_of_bytes))
    game_file.write(bytes.getvalue())

def convert_numpy_array(numpy_array):
    """
    Converts a numpy array into compressed bytes
    :param numpy_array: An array that is going to be converted into bytes
    :return: A BytesIO object that contains compressed bytes
    """
    compressed_array = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
    np.save(compressed_array, numpy_array)
    return compressed_array


def write_version_info(file, version_number):
    file.write(struct.pack('i', version_number))


def get_file_version(file):
    dir_name = os.path.basename(os.path.dirname(file.name))
    time = dir_name.split('-')[0]
    print('time: ' + time)
    if int(time) < 1512069505446:
        return NON_FLIPPED_FILE_VERSION
    try:
        chunk = file.read(4)
        file_version = struct.unpack('i', chunk)[0]
        return str(file_version)
    except:
        print('file was empty')
        return EMPTY_FILE


def read_data(file, process_pair_function):
    """
    Reads a file.  Quits if anything breaks.
    :param file: A simple python file object that will be read
    :param process_pair_function: A function that takes in an input array and an output array.
    There is also an optional number saying how many times this has been called for a single file.
    It always starts at 0
    :return: None
    """

    file_version = get_file_version(file)
    if file_version == EMPTY_FILE:
        return

    print('replay version:', file_version)

    pair_number = 0
    totalbytes = 0
    while True:
        try:
            chunk = file.read(4)
            if chunk == '':
                totalbytes += 4
                break
            input_array, num_bytes = get_array(file, chunk)
            totalbytes += num_bytes + 4
            chunk = file.read(4)
            if chunk == '':
                totalbytes += 4
                break
            output_array, num_bytes = get_array(file, chunk)
            process_pair_function(input_array, output_array, pair_number, file_version)
            pair_number += 1
            totalbytes += num_bytes + 4
        except EOFError:
            print('reached end of file')
            break
    file_size = os.stat(file.name).st_size
    if file_size - totalbytes <= 4:
        print('read: 100% of file')
    else:
        print('read: ' + str(totalbytes) + '/' + str() + ' bytes')


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
    return result, starting_byte


def default_process_pair(input_array, output_array, pair_number):
    pass


if __name__ == '__main__':
    test_file = open('training\\1511676262748-30\\SaltieRl(2).txt', 'r+b')
    print(test_file)
    read_data(test_file, default_process_pair)
