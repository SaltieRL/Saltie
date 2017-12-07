import io
import os
import sys
import struct
import hashlib

import numpy as np

EMPTY_FILE = 'empty'
NON_FLIPPED_FILE_VERSION = 0
FLIPPED_FILE_VERSION = 1
HASHED_NAME_FILE_VERSION = 2

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


def write_bot_name(game_file, name):
    hashed_name = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 2 ** 64
    print('hashed_name', hashed_name)
    game_file.write(struct.pack('Q', hashed_name))


def get_file_version(file):
    file_name = 0
    if not isinstance(file, io.BytesIO):
        file_name = os.path.basename(file.name).split('-')[0]

    try:
        chunk = file.read(4)
        file_version = struct.unpack('i', chunk)[0]
        if file_version < HASHED_NAME_FILE_VERSION:
            return str(file_version), file_name
        else:
            chunk = file.read(8)
            hashed_name = struct.unpack('Q', chunk)[0]
            return file_version, str(hashed_name)
    except:
        print('file was empty', sys.exc_info()[0])
        return EMPTY_FILE, file_name


def get_file_size(f):
    # f is a file-like object.
    try:
        old_file_position = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(old_file_position, os.SEEK_SET)
        return size
    except:
        return 0


def read_data(file, process_pair_function):
    """
    Reads a file.  Quits if anything breaks.
    :param file: A simple python file object that will be read
    :param process_pair_function: A function that takes in an input array and an output array.
    There is also an optional number saying how many times this has been called for a single file.
    It always starts at 0
    :return: None
    """

    file_version, hashed_name = get_file_version(file)
    if file_version == EMPTY_FILE:
        return

    print('replay version:', file_version)
    print('hashed name:', hashed_name)

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
            process_pair_function(input_array, output_array, pair_number, hashed_name)
            pair_number += 1
            totalbytes += num_bytes + 4
        except EOFError:
            print('reached end of file')
            break
    file_size = get_file_size(file)
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

