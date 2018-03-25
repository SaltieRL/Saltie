from Procedure import pre_process, feedback, gather_info
from Handling import controls
from Strategy import strategy


def Process(s, game, version=3):

    pre_process(s, game)
    gather_info(s)
    strategy(s)
    controls(s)
    feedback(s)

    return output(s, version)


def output(s, version):
    return [s.throttle, s.steer, s.pitch, s.yaw, s.roll, s.jump, s.boost, s.powerslide]
