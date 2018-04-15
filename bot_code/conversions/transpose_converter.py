from bot_code.conversions import binary_converter
import numpy as np
import os

def get_pair_byte_length(file_version):
    return len(pair_to_bytes(
        np.zeros((binary_converter.get_state_dim(file_version))),
        np.zeros((binary_converter.get_output_vector_dim(file_version)))
   ))

def pair_to_bytes(state_array, output_vector_array):
    return binary_converter.numpy_array_to_bytes(
        np.concatenate((state_array, output_vector_array))
    )

def transpose_replay_file(input_file, output_file):
    file_version, hashed_name, is_eval = binary_converter.read_header(input_file)
    assert file_version == binary_converter.get_latest_file_version()
    input_file.seek(0)

    rows_of_tick_bytes = []  # one row per game tick.
    state_dim = binary_converter.get_state_dim(file_version)
    output_vector_dim = binary_converter.get_output_vector_dim(file_version)
    expected_pair_length = get_pair_byte_length(file_version)
    for state_array, output_vector_array, pair_number in binary_converter.iterate_data(input_file, batching=False, verbose=False):
        assert len(state_array) == state_dim
        assert len(output_vector_array) == output_vector_dim
        tick_bytes = pair_to_bytes(state_array, output_vector_array)
        assert len(tick_bytes) == expected_pair_length
        rows_of_tick_bytes.append(tick_bytes)

    binary_converter.write_header_to_file(output_file, hashed_name, is_eval)
    write_transposed_to_file(output_file, rows_of_tick_bytes)

def untranspose_replay_file(input_file, output_file, batch_size=1000):
    file_version, hashed_name, is_eval = binary_converter.read_header(input_file)
    assert file_version == binary_converter.get_latest_file_version()
    pair_length = get_pair_byte_length(file_version)

    # Figure out the column length
    after_header_file_position = input_file.tell()
    input_file.seek(0, os.SEEK_END)
    total_transposed_bytes = input_file.tell() - after_header_file_position
    input_file.seek(after_header_file_position, os.SEEK_SET)
    col_length, mod = divmod(total_transposed_bytes, pair_length)
    assert mod == 0, "file length must be a multiple of the pair_length"

    cols = []
    while True:
        try:
            col = input_file.read(col_length)
        except EOFError:
            break
        if len(col) == 0:
            break
        assert len(col) == col_length
        cols.append(col)

    state_dim = binary_converter.get_state_dim(file_version)
    output_vector_dim = binary_converter.get_output_vector_dim(file_version)
    states = []
    output_vectors = []
    for row in zip(*cols):
        pair = binary_converter.to_numpy_array(bytes(row))
        assert pair.shape == (state_dim + output_vector_dim,)
        states.append(pair[:state_dim])
        output_vectors.append(pair[state_dim:])

    binary_converter.write_header_to_file(output_file, hashed_name, is_eval)
    for state_batch, output_vector_batch in zip(batches(states, batch_size), batches(output_vectors, batch_size)):
        binary_converter.write_array_to_file(output_file, np.array(state_batch).flatten())
        binary_converter.write_array_to_file(output_file, np.array(output_vector_batch).flatten())
    # binary_converter.

def batches(l, n):
    """Yield successive n-sized (or smaller) batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Above are asymmetric replay-specific functions
# Below are general purpose symmetric funcitons

def transpose_file_with_header(input_file, output_file, header_length, row_length):
    '''
    takes a saltie replay and makes it easier to compress.
    handles the header and forwards to transpose_file for the
    :param header_length: is the number of bytes the header takes up at the start of the file
    :param row_length: is the number of bytes each row / chunk-pair / tick takes up.
        Rows are assumed to come after the header until the end of the file.
    '''
    header = input_file.read(header_length)
    assert len(header) == header_length
    output_file.write(header)
    transpose_file(input_file, output_file, row_length)

def transpose_file(input_file, output_file, row_length):
    '''
    :param input_file: is the binary-file object we'll read.
        It's contents should look like [row][row][row]... until the end of the file.
        If your file has a header, you may seek to the start of this repeated-struct block.
    :param output_file: is the file-like object that we'll write to
    :param row_length: is the length of rows in the input file, measures in bytes.
    :param row_length: is the number of bytes each row / chunk-pair / tick takes up.
        Rows are assumed to come after the header until the end of the file.
    '''
    rows = []
    while True:
        row = input_file.read(row_length)
        if len(row) == 0:
            break
        # TODO: Throw nicer exceptions if we want to reuse this.
        assert len(row) == row_length, "File seems corrupted. It ended with a partial row."
        rows.append(row)

    write_transposed_to_file(output_file, rows)

def write_transposed_to_file(output_file, rows_of_bytes):
    for col in zip(*rows_of_bytes):
        output_file.write(bytes(col))
