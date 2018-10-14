# MIT License
#
# Copyright (c) 2018 LHolten@Github Hytak#5125@Discord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy
import math


class Atba:
    def __init__(self):
        numpy.seterr(all='raise')

    def get_action(self, arr):
        spatial = arr[0][0]

        action = numpy.zeros(9)

        left_vector = spatial[:, 7]
        forward_vector = spatial[:, 6]
        up_vector = spatial[:, 8]
        car_location = spatial[:, 0]
        ball_location = spatial[:, 1]
        goal_location = numpy.array([0, 5.12, 0.3])
        own_goal_location = numpy.array([0, -5.12, 0.3])

        relative_ball = ball_location - car_location
        ball_distance = numpy.linalg.norm(relative_ball)
        relative_ball /= ball_distance
        relative_goal = goal_location - car_location
        relative_goal /= numpy.linalg.norm(relative_goal)
        relative_own_goal = own_goal_location - car_location
        relative_own_goal /= numpy.linalg.norm(relative_own_goal)

        # offence/ defence switching

        offence = (1 + ball_location[1] / 5.12) / 2
        defence = 1 - offence

        ball_direction = (1 + offence * relative_ball @ relative_goal - defence * relative_ball @ relative_own_goal)/2
        ball_direction = pow(ball_direction, 1.4)
        not_ball_direction = 1 - ball_direction

        car_offence = (1 + car_location[1] / 5.12) / 2
        car_defence = 1 - car_offence

        # controls

        left_ball = relative_ball @ left_vector
        left_opp_goal = relative_goal @ left_vector
        left_own_goal = relative_own_goal @ left_vector
        left_goal = car_defence * left_own_goal - car_offence * left_opp_goal
        steer = ball_direction * left_ball + not_ball_direction * left_goal
        roll = ball_direction * left_goal + not_ball_direction * -left_vector[2]

        forward_ball = relative_ball @ forward_vector
        forward_opp_goal = relative_goal @ forward_vector
        forward_own_goal = relative_own_goal @ forward_vector
        forward_goal = car_defence * forward_own_goal - car_offence * forward_opp_goal
        pitch = ball_direction * forward_goal + not_ball_direction * forward_vector[2]

        up_ball = relative_ball @ up_vector
        jump = ball_direction * up_ball + not_ball_direction * -1
        throttle = ball_direction * math.copysign(pow(1 - abs(up_ball), 6), forward_ball) + not_ball_direction * 1

        action[0] = throttle
        action[1] = math.copysign(pow(abs(pitch), 2), pitch)
        action[2] = 1 if throttle > 0.75 else -1
        action[3] = 1 if abs(steer) > 0.65 else -1
        action[4] = jump if ball_distance < 0.4 else -1
        action[5] = 1 if ball_distance < 0.3 else -1
        action[6] = math.copysign(pow(abs(steer), 0.1), steer)
        action[7] = math.copysign(pow(abs(steer), 0.1), steer)
        action[8] = math.copysign(pow(abs(roll), 2), -roll)

        return numpy.array([action])
