from Procedure import pre_process, feedback, gather_info
from Handling import controls
from Strategy import strategy
from Util import U
# import time


def Process(s, game, version=3):

    # t0 = time.time()

    pre_process(s, game)
    gather_info(s)
    strategy(s)
    controls(s)
    feedback(s)

    # if not s.counter % 50:
    #     print(1 / 60 - (time.time() - t0))

    return output(s, version)


def output(s, version):
    if version == 2:

        if s.roll != 0 and s.counter % 3 == 1 and abs(s.r) > .04:
            s.yaw = s.roll
            s.powerslide = 1

        if s.poG:
            s.yaw = s.steer

        return [int((s.yaw + 1) * U / 2), int((s.pitch + 1) * U / 2), int(s.throttle * U),
                int(-s.throttle * U), s.jump, s.boost, s.powerslide]

    else:

        return [s.throttle, s.steer, s.pitch, s.yaw, s.roll, s.jump, s.boost, s.powerslide]
