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


class SpatialInput(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.location = nn.Linear(2, size, bias=True)
        self.velocity = nn.Linear(2, size, bias=True)
        self.angular_velocity = nn.Linear(2, size, bias=True)
        self.normal = nn.Linear(3, size, bias=False)

    def forward(self, spatial):
        processed_location = self.location(spatial[:, 0:2])
        processed_velocity = self.velocity(spatial[:, 2:4])
        processed_angular_velocity = self.angular_velocity(spatial[:, 4:6])
        processed_normal = self.normal(spatial[:, 6:9])

        return processed_location * processed_velocity * processed_angular_velocity * processed_normal


class ActorModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.input_x = SpatialInput(10)
        self.input_y = SpatialInput(10)
        self.input_z = SpatialInput(10)

        self.linear = nn.Linear(30, 9)

    def forward(self, spatial, car_stats):
        processed_x = self.input_x(spatial[:, 0])
        processed_y = self.input_y(spatial[:, 1])
        processed_z = self.input_z(spatial[:, 2])

        processed_spatial = torch.cat([processed_x, processed_y, processed_z], dim=1)

        return self.linear(processed_spatial)


class SymmetricModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.actor = ActorModel()
        self.tanh = nn.Tanh()

    def forward(self, own_car_stats,
                own_team_car_stats,
                opp_team_car_stars,
                own_car_spatial,
                own_team_car_spatial,
                opp_team_car_spatial,
                game_ball_spatial):
        spatial_inv = torch.tensor(spatial)
        spatial_inv[:, 0] *= -1  # invert x coordinates
        spatial_inv[:, :, 7] *= -1  # invert own car left normal
        spatial_inv[:, :, 4:6] *= -1  # invert angular velocity

        output = self.actor(spatial, car_stats)
        output_inv = self.actor(spatial_inv, car_stats)

        output[:, 0:6] += output_inv[:, 0:6]  # combine unflippable outputs
        output[:, 6:9] += -1 * output_inv[:, 6:9]  # combine flippable outputs

        output = self.tanh(output)

        return output
