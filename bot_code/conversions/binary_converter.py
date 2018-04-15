import io
import os
import struct

import numpy as np
import time
import logging
from bot_code.conversions.input import input_formatter

import gzip

EMPTY_FILE = 'empty'
NO_FILE_VERSION = -1
NON_FLIPPED_FILE_VERSION = 0
FLIPPED_FILE_VERSION = 1
HASHED_NAME_FILE_VERSION = 2
IS_EVAL_FILE_VERSION = 3
BATCH_ARRAY_FILE_VERSION = 4
TIME_ADDITION_FILE_VERSION = 5


def get_latest_file_version():
    return TIME_ADDITION_FILE_VERSION

def get_state_dim(file_version):
    if file_version == 4:
        return 206
    elif file_version is get_latest_file_version():
        return input_formatter.get_state_dim()

def get_output_vector_dim(file_version):
    return 8

def write_header_to_file(game_file, bot_hash, is_eval):
    write_version_info(game_file, get_latest_file_version())
    write_bot_hash(game_file, bot_hash)
    write_is_eval(game_file, is_eval)


def write_array_to_file(game_file, array):
    """
    :param game_file: This is the file that the array will be written to.
    :param array: A numpy array of any size.
    """
    bytes = numpy_array_to_bytes(array)
    game_file.write(struct.pack('i', len(bytes)))
    game_file.write(bytes)

def numpy_array_to_bytes(numpy_array):
    """
    Converts a numpy array into bytes
    :param numpy_array: An array that is going to be converted into bytes
    :return: A BytesIO object that contains compressed bytes
    """
    compressed_array = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
    np.save(compressed_array, numpy_array, allow_pickle=False, fix_imports=False)
    return compressed_array.getvalue()

def write_version_info(file, version_number):
    file.write(struct.pack('i', version_number))


def write_bot_hash(game_file, hashed_name):
    game_file.write(struct.pack('Q', hashed_name))


def write_is_eval(game_file, is_eval):
    game_file.write(struct.pack('?', is_eval))


def read_header(file):
    """
    Gets file info from the file
    :param file:
    :return: a tuple containing
            file_version:  This is the version of a file represented as a number.
            hashed_name: This is the hash of the model that was used to create this file.  If it is a least version 2
            is_eval: This is used to decide if the file was created in eval mode
    """
    if not isinstance(file, io.BytesIO):
        file_name = os.path.basename(file.name).split('-')[0]
    else:
        file_name = 'ram'

    result = []

    try:
        chunk = file.read(4)
        file_version = struct.unpack('i', chunk)[0]
        if file_version > get_latest_file_version():
            file.seek(0, 0)
            file_version = NO_FILE_VERSION

        result.append(file_version)

        if file_version < HASHED_NAME_FILE_VERSION:
            result.append(file_name)
        else:
            chunk = file.read(8)
            hashed_name = struct.unpack('Q', chunk)[0]
            result.append(hashed_name)
        if file_version < IS_EVAL_FILE_VERSION:
            result.append(False)
        else:
            chunk = file.read(1)
            is_eval = struct.unpack('?', chunk)[0]
            result.append(is_eval)
    except Exception as e:
        result = [EMPTY_FILE, file_name, False]
        print('file version was messed up', e)
    finally:
        return tuple(result)


def get_file_size(f):
    """
    :param f: A file-like object that should support seeking to the end.
    :return: The size of the file in bytes.
    """
    try:
        old_file_position = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(old_file_position, os.SEEK_SET)
        return size
    except:
        return 0


def read_data(file, process_pair_function, batching=False):
    """
    Reads a file and does a callback to process_pair_function.
    Quits if anything breaks.

    :param file: A simple python file object that will be read
    :param process_pair_function: A function that takes in an input array and an output array.
    The following may be out of date:
        There is also an optional number saying how many times this has been called for a single file.
        It always starts at 0
    :param batching: If more than one item in an array is read at the same time then we will batch
    them instead of doing them one at a time
    """
    file_version, hashed_name, is_eval = read_header(file)
    file.seek(0)
    for state_array, output_vector_array, pair_number in iterate_data(file, batching=batching):
        process_pair_function(state_array, output_vector_array, pair_number, hashed_name)


def iterate_data(file, batching=False, verbose=True):
    """
    Reads a saltie replay file and returns its data as an iterator.
    Quits if anything breaks.
    :param file: A binary python file object that will be read
    :param batching: If more than one item in an array is read at the same time then we will batch
    them instead of doing them one at a time

    :return: Iterator<(state_array, output_vector_array, pair_number)>
        state_array:
            A list of floats representing the game state. (game_tick_packet)
            Produced by input_formatter.py.
            Can be read with
        output_vector_array:
            A list of floats representing the controller.
            Should be analogous to what a RLBot agent returns in get_output_vector().
        pair_number:
            The index of the state_array/output_vector_array pair.
            Increments by 1 if not batching, may increment in larger strides when batching.
    """

    file_version, hashed_name, is_eval = read_header(file)
    if file_version == EMPTY_FILE:
        return

    if verbose:
        print('Reading replay with version:', file_version)
        # print('hashed name:', hashed_name)

    if file_version is 4:
        upgrade_state_array = v4tov5
    else:
        upgrade_state_array = lambda x: x

    pair_number = 0
    total_external_time = 0
    total_numpy_conversion_time = 0
    start_time = time.clock()
    for state_bytes, output_vector_bytes in as_non_overlapping_pairs(iterate_chunks(file)):
        pair_start_time = time.clock()

        # Convert to numpy arrays
        state_array = to_numpy_array(state_bytes)
        output_vector_array = to_numpy_array(output_vector_bytes)
        chunk_start_time = time.clock()
        batch_size = int(len(state_array) / get_state_dim(file_version))
        state_array = np.reshape(state_array, (batch_size, int(get_state_dim(file_version))))
        output_vector_array = np.reshape(output_vector_array, (batch_size, get_output_vector_dim(file_version)))
        numpy_end_time = time.clock()

        # External processing
        if not batching:
            for i in range(len(state_array)):
                yield (upgrade_state_array(state_array[i]), output_vector_array[i], pair_number)
                pair_number += 1
        else:
            yield (upgrade_state_array(state_array), output_vector_array, pair_number)
            pair_number += batch_size

        pair_end_time = time.clock()
        total_numpy_conversion_time = numpy_end_time - pair_start_time
        total_external_time += pair_end_time - numpy_end_time

    # Print some stats
    if verbose:
        total_time = time.clock() - start_time
        print('total pairs: {}'.format(pair_number))
        print('time reading chunks:        {:.05f}s'.format(total_time - (total_numpy_conversion_time + total_external_time)))
        print('time converting to numpy:   {:.05f}s'.format(total_numpy_conversion_time))
        print('time externally processing: {:.05f}s'.format(total_external_time))

def v4tov5(state_array):
    # Passed time (after game_info) 1
    state_array = np.insert(state_array, 1, 0.0, axis=1)
    for i in range(6):
        i = 22 if i is 0 else 43 + 20 * i
        state_array = np.insert(state_array, i, 0, axis=1)
        state_array = np.insert(state_array, i + 1, np.greater(np.hypot(np.hypot(state_array[:, i - 6], state_array[:, i - 5]), state_array[:, i - 4]), 2200), axis=1)
    return state_array

def as_non_overlapping_pairs(iterable):
    """
    [1,2,3,4,5,6] -> [(1,2), (3,4), (5,6)]
    The given iterable must be of even length.
    """
    first = None  # the first item in the tuple
    i = -1
    for i, item in enumerate(iterable):
        if i % 2 == 0:
            first = item
        else:
            yield (first, item)
    if i % 2 == 0:
        raise Exception('Missing the second item of the pair. i=' + str(i))

def iterate_chunks(file):
    """
    Reads chunks until the end of the file, yielding them one by one
    :param file: A binary python file object that will be read.
        The file from the current position needs to have the following format:
        [chunk_size][chunk][chunk_size][chunk]...
        where chunk_size is a signed int32 representing the number of bytes in the following chunk.

    :return: Iterator<bytes>
    """
    while True:
        try:
            chunk_size_byte_string = file.read(4)
        except EOFError:
            # print('reached end of file')
            break


        if len(chunk_size_byte_string) == 0:
            break
        elif len(chunk_size_byte_string) == 4:
            chunk_size = to_int(chunk_size_byte_string)
        else:
            raise EOFError('chunk_size was truncated')

        def raise_bad_chunk_size(actual_size):
            raise EOFError('The file promised there were {} bytes in the chunk but there were {}'.format(chunk_size, actual_size))
        try:
            chunk = file.read(chunk_size)
        except EOFError:
            raise_bad_chunk_size(0)
        if len(chunk) != chunk_size:
            raise_bad_chunk_size(len(chunk))

        yield chunk

def to_int(byte_string):
    assert len(byte_string) == 4
    return struct.unpack('i', byte_string)[0]

def to_numpy_array(chunk):
    fake_file = io.BytesIO(chunk)
    try:
        result = np.load(fake_file, fix_imports=False)
    except OSError:
        print('numpy parse error')
        raise EOFError
    return result


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
        #print('struct error')
        raise EOFError
    numpy_bytes = file.read(starting_byte)
    return to_numpy_array(numpy_bytes), starting_byte


def print_values(state_array, output_vector_array, somevalue, anothervalue):
    return

if __name__ ==  '__main__':
    with gzip.open("path_to_file", 'rb') as f:
        try:
            read_data(f, print_values, batching=True)
        except Exception as e:
            print('error training on file ', e)
