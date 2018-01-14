import numpy as np

from conversions.input.input_formatter import InputFormatter


class SimpleInputFormatter(InputFormatter):

    def create_input_array(self, game_tick_packet, passed_time=0.0):
        # posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt, ballx, bally, ballz, ballvx, ballvy, ballvz
        inputs = [game_tick_packet.gamecars[self.index].Location.X,
                  game_tick_packet.gamecars[self.index].Location.Y,
                  game_tick_packet.gamecars[self.index].Location.Z,
                  game_tick_packet.gamecars[self.index].Rotation.Pitch,
                  game_tick_packet.gamecars[self.index].Rotation.Yaw,
                  game_tick_packet.gamecars[self.index].Rotation.Roll,
                  game_tick_packet.gamecars[self.index].Velocity.X,
                  game_tick_packet.gamecars[self.index].Velocity.Y,
                  game_tick_packet.gamecars[self.index].Velocity.Z,
                  game_tick_packet.gamecars[self.index].AngularVelocity.X,
                  game_tick_packet.gamecars[self.index].AngularVelocity.Y,
                  game_tick_packet.gamecars[self.index].AngularVelocity.Z,
                  game_tick_packet.gamecars[self.index].Boost,
                  game_tick_packet.gameball.Location.X,
                  game_tick_packet.gameball.Location.Y,
                  game_tick_packet.gameball.Location.Z,
                  game_tick_packet.gameball.Velocity.X,
                  game_tick_packet.gameball.Velocity.Y,
                  game_tick_packet.gameball.Velocity.Z
                  ]
        inputs = np.array(inputs).reshape((1, -1))
        return inputs
