



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
        It's contents should look like [struct][struct][struct]... until the end of the file.
        Where the structs are of length @struct_bytes.
        If your file has a header, you may seek to the start of this repeated-struct block.
    :param output_file: is the file-like object that we'll write to
    :param row_length: is the number of bytes each row / chunk-pair / tick takes up.
        Rows are assumed to come after the header until the end of the file.
    '''
    structs = []
    while True:
        struct = input_file.read(row_length)
        if len(struct) == 0:
            break
        # TODO: Throw nicer exceptions if we want to reuse this.
        assert len(struct) == row_length, "File seems corrupted. It ended with a partial row."
        structs.append(struct)

    for col in zip(*structs):
        output_file.write(bytes(col))

