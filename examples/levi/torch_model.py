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
        self.angular = nn.Linear(2, size, bias=True)
        self.normal = nn.Linear(3, size, bias=False)

    def forward(self, own_car_axis, game_ball_axis):
        processed_location = self.location(torch.stack((own_car_axis[:, 0], game_ball_axis[:, 0]), dim=1))
        processed_velocity = self.velocity(torch.stack((own_car_axis[:, 1], game_ball_axis[:, 1]), dim=1))
        processed_angular = self.angular(torch.stack((own_car_axis[:, 2], game_ball_axis[:, 2]), dim=1))
        processed_normal = self.normal(own_car_axis[:, 3:6])

        return processed_location * processed_velocity * processed_angular * processed_normal


class ActorModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.input_x = SpatialInput(10)
        self.input_y = SpatialInput(10)
        self.input_z = SpatialInput(10)

        self.linear = nn.Linear(30, 9)

    def forward(self, own_car_spatial, game_ball_spatial):
        processed_x = self.input_x(own_car_spatial[:, 0], game_ball_spatial[:, 0])
        processed_y = self.input_y(own_car_spatial[:, 1], game_ball_spatial[:, 1])
        processed_z = self.input_z(own_car_spatial[:, 2], game_ball_spatial[:, 2])

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

        own_car_spatial_inv = torch.tensor(own_car_spatial)
        own_car_spatial_inv[:, 0] *= -1  # invert x coordinates
        own_car_spatial_inv[:, :, 4] *= -1  # invert left normal
        own_car_spatial_inv[:, :, 2] *= -1  # invert angular velocity

        game_ball_spatial_inv = torch.tensor(game_ball_spatial)
        game_ball_spatial_inv[:, 0] *= -1  # invert x coordinates
        game_ball_spatial_inv[:, :, 2] *= -1  # invert angular velocity

        output = self.actor(own_car_spatial, game_ball_spatial)
        output_inv = self.actor(own_car_spatial_inv, game_ball_spatial_inv)

        output[:, 0:6] += output_inv[:, 0:6]  # combine unflippable outputs
        output[:, 6:9] += -1 * output_inv[:, 6:9]  # combine flippable outputs

        output = self.tanh(output)

        return output
