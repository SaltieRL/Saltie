import ctypes
import mmap
import numpy as np

BUFFER_SIZE = 100
LAST_INDEX = BUFFER_SIZE - 1
USED_BUFFER_FOR_GRAPHING = 100
SHARED_MEMORY_PREFIX = 'Local\\RLBotLiveDataPlayer'


class RotatingBufferStruct(ctypes.Structure):
    _fields_ = [("cur_index", ctypes.c_int),
                ("expected_rewards", ctypes.c_float * BUFFER_SIZE)]


# A rotating buffer backed by shared memory so it can be used across processes.
class RotatingBuffer:
    def __init__(self, index):
        self.buff = mmap.mmap(-1, ctypes.sizeof(RotatingBufferStruct), SHARED_MEMORY_PREFIX + str(index))
        self.rot_buf = RotatingBufferStruct.from_buffer(self.buff)

    def __iadd__(self, other):
        next_index = (self.rot_buf.cur_index + 1) % BUFFER_SIZE
        self.rot_buf.expected_rewards[next_index] = other
        self.rot_buf.cur_index = next_index
        return self

    def print_rotating_buffer(self):
        print(self.rot_buf.cur_index)
        print(np.array(self.rot_buf.expected_rewards))

    def get_current_buffer(self):
        buffer = np.array(self.rot_buf.expected_rewards)
        first_slice = np.flipud(buffer[max(0,self.rot_buf.cur_index - USED_BUFFER_FOR_GRAPHING):self.rot_buf.cur_index])
        second_slice = np.flipud(buffer[(-1 * (USED_BUFFER_FOR_GRAPHING - first_slice.size)):])
        return np.append(first_slice,second_slice)

# Testing
if __name__ == '__main__':
    rb = RotatingBuffer(0)
    rb.print_rotating_buffer()
    rb += 3.7
    rb += 4.7
    rb.print_rotating_buffer()
    print(rb.get_current_buffer())




