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

import torch
import torch.nn as nn


# not used
class SpatialInput(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.location = nn.Linear(2, size, bias=True)
        self.velocity = nn.Linear(2, size, bias=True)
        self.angular_velocity = nn.Linear(2, size, bias=True)
        self.normal = nn.Linear(3, size, bias=True)

    def forward(self, spatial):
        processed_location = self.location(spatial[:, 0:2])
        processed_velocity = self.velocity(spatial[:, 2:4])
        processed_angular_velocity = self.angular_velocity(spatial[:, 4:6])
        processed_normal = self.normal(spatial[:, 6:9])

        return processed_location * processed_velocity * processed_angular_velocity * processed_normal


class SimpleSpatialInput(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.multiplier = nn.Linear(6, size, bias=True)
        self.normal = nn.Linear(3, size, bias=True)

    def forward(self, spatial):
        processed_multiplier = self.multiplier(spatial[:, 0:6])
        processed_normal = self.normal(spatial[:, 6:9])

        return processed_multiplier * processed_normal


class ActorModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.input_x = SimpleSpatialInput(10)
        self.input_y = SimpleSpatialInput(10)
        self.input_z = SimpleSpatialInput(5)

        self.linear = nn.Linear(25, 25, bias=True)
        self.soft_sign = nn.Softsign()
        self.output = nn.Linear(25, 15, bias=True)

    def forward(self, spatial, car_stats):
        processed_x = self.input_x(spatial[:, 0])
        processed_y = self.input_y(spatial[:, 1])
        processed_z = self.input_z(spatial[:, 2])

        processed_spatial = torch.cat([processed_x, processed_y, processed_z], dim=1)
        result = self.linear(processed_spatial)
        result = self.soft_sign(result)

        return self.output(result)


# not used
class CombinedActorModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.softmax = nn.Softmax(dim=1)
        self.actor_list = []
        for i in range(3):
            m = ActorModel()
            self.actor_list.append(m)
            self.add_module('actor' + str(i), m)

    def forward(self, spatial, car_stats):
        result = torch.stack([self.actor_list[i](spatial, car_stats) for i in range(len(self.actor_list))], dim=2)

        multiplier = self.softmax(result[:, 9, :])

        result2 = torch.stack([result[:, i] * multiplier for i in range(9)], dim=1)

        result3 = torch.cumsum(result2, dim=2)

        result3 = result3[:, :, len(self.actor_list) - 1]

        return result3


class SymmetricModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.actor = ActorModel()
        self.soft_sign = nn.Softsign()
        self.soft_plus = nn.Softplus(beta=1, threshold=20)

        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, spatial, car_stats):
        spatial_inv = spatial.clone().detach()
        spatial_inv[:, 0] *= -1  # invert x coordinates
        spatial_inv[:, :, 7] *= -1  # invert own car left normal
        spatial_inv[:, :, 4:6] *= -1  # invert angular velocity

        output = self.actor(spatial, car_stats)
        output_inv = self.actor(spatial_inv, car_stats)

        output[:, 0:9] += output_inv[:, 0:9]  # combine unflippable outputs
        output[:, 9:13] += -1 * output_inv[:, 9:13]  # combine flippable outputs

        output[:, 13:15] += output_inv[:, 13:15]  # combine the values for the time estimate

        controls = self.soft_sign(output[:, 0:13])

        time_estimate = self.soft_plus(output[:, 13]) / self.soft_plus(output[:, 14])
        # print(time_estimate, self.soft_plus(self.scale * time_estimate))
        time_distribution = torch.distributions.Normal(time_estimate, self.soft_plus(self.scale) * time_estimate, True)

        return controls, time_distribution

    @staticmethod
    def get_input_state_dimension():
        return [(3, 9), (5,)]

    @staticmethod
    def get_model_output_dimension():
        return [(13,), (1,)]
