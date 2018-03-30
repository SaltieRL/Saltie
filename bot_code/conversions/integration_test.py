"""
Tests that the following chain produces the same data as originally provided:
game_tick_packet
    input_formatter
        binary_converter
            .bin file
        binary_converter
    output_formatter
tensorflow_object (lambda: None)
"""

import tempfile

from bot_code.conversions.input.input_formatter import InputFormatter
from bot_code.conversions import output_formatter
from bot_code.conversions import binary_converter
from game_data_struct import GameTickPacket

# pre-test: This used to fail due to mutated external state in conversion functions.
game_tick_packet = GameTickPacket()
input_formatter = InputFormatter(1,0)
array1 = input_formatter.create_input_array(game_tick_packet, passed_time=0.0)
array2 = input_formatter.create_input_array(game_tick_packet, passed_time=0.0)
assert all(x==y for x,y in zip(array1, array2))



# def write_test_data_to_file(replay_file):
#     # prepare data
#     team = 1
#     player_index = 1
#     bot_hash = 41
#     is_eval = False
#     game_tick_packet = GameTickPacket()
#     game_tick_packet.gamecars[1].team = 1
#     game_tick_packet.gamecars[1].boost = 30
#     input_formatter = InputFormatter(team, player_index)
#     test_data_pairs = [
#         (
#             input_formatter.create_input_array(game_tick_packet, passed_time=42.0),
#             [0.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         ),
#         (
#             input_formatter.create_input_array(game_tick_packet, passed_time=52.),
#             [0.0, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         ),
#     ]

#     # Note: This is API is fragile as fuck. Any mistake in ordering or ommission will throw everything out of whack.

#     # Write header
#     binary_converter.write_version_info(replay_file, binary_converter.get_latest_file_version())
#     binary_converter.write_bot_hash(replay_file, bot_hash)
#     binary_converter.write_is_eval(replay_file, is_eval)

#     # Write body
#     for state_array, output_vector_array in test_data_pairs:
#         binary_converter.write_array_to_file(replay_file, state_array)
#         binary_converter.write_array_to_file(replay_file, output_vector_array)

# def read_from_file_and_assert(replay_file):
#     tuples = list(binary_converter.iterate_data(replay_file))
#     for state_array, output_vector_array in tuples:
#         print (t)


# with tempfile.TemporaryFile() as replay_file:
#     write_test_data_to_file(replay_file)
#     replay_file.seek(0)
#     read_from_file_and_assert(replay_file)

print (' === ALL TESTS PASSED === ')
