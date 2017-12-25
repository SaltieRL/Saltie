import math
import time


class Agent:
    def __init__(self, name, team, index):
        self.index = index

        # Contants
        self.DODGE_TIME = 0.2
        self.DISTANCE_TO_DODGE = 500
        self.DISTANCE_FROM_BALL_TO_BOOST = 1500  # The minimum distance the ball needs to be away from the bot for the bot to boost
        self.POWERSLIDE_ANGLE = 170  # The angle (from the front of the bot to the ball) at which the bot should start to powerslide.

        # Controller inputs
        self.throttle = 0
        self.steer = 0
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        self.boost = False
        self.jump = False
        self.powerslide = False

        # Game values
        self.bot_pos = None
        self.bot_rot = None
        self.ball_pos = None
        self.bot_yaw = None

        # Dodging
        # self.should_dodge = False
        # self.on_second_jump = False
        # self.next_dodge_time = 0

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def aim(self, target_x, target_y):
        angle_between_bot_and_target = math.degrees(math.atan2(target_y - self.bot_pos.Y,
                                                               target_x - self.bot_pos.X))

        angle_front_to_target = angle_between_bot_and_target - self.bot_yaw

        # Correct the values
        if angle_front_to_target < -180:
            angle_front_to_target += 360
        if angle_front_to_target > 180:
            angle_front_to_target -= 360

        if angle_front_to_target < -10:
            # If the target is more than 10 degrees right from the centre, steer left
            self.steer = -1
        elif angle_front_to_target > 10:
            # If the target is more than 10 degrees left from the centre, steer right
            self.steer = 1
        else:
            # If the target is less than 10 degrees from the centre, steer straight
            self.steer = 0

        if abs(math.degrees(angle_front_to_target)) < self.POWERSLIDE_ANGLE:
            self.powerslide = True
        else:
            self.powerslide = False

    # def check_for_dodge(self, target_x, target_y, values):
    #     if self.distance(self.bot_pos.X, self.bot_pos.Y, self.ball_pos.X,
    #                      self.ball_pos.Y) < self.DISTANCE_TO_DODGE and not values.gamecars[0].bDoubleJumped:
    #         self.aim(target_x, target_y)
    #         self.jump = True
    #         self.pitch = -1

    def get_output_vector(self, values):
        # Update game data variables
        self.bot_pos = values.gamecars[self.index].Location
        self.bot_rot = values.gamecars[self.index].Rotation
        self.ball_pos = values.gameball.Location

        # This sets self.jump to be active for only 1 frame
        self.jump = False

        # Get car's yaw and convert from Unreal Rotator units to degrees
        self.bot_yaw = abs(self.bot_rot.Yaw) % 65536 / 65536 * 360
        if self.bot_rot.Yaw < 0:
            self.bot_yaw *= -1

        # Boost when ball is far enough away
        if self.distance(self.bot_pos.X, self.bot_pos.Y, self.ball_pos.X,
                         self.ball_pos.Y) > self.DISTANCE_FROM_BALL_TO_BOOST:
            self.boost = True
        else:
            self.boost = False

        # Blue has their goal at -5000 (Y axis) and orange has their goal at 5000 (Y axis). This means that:
        # - Blue is behind the ball if the ball's Y axis is greater than blue's Y axis
        # - Orange is behind the ball if the ball's Y axis is smaller than orange's Y axis
        self.throttle = 1
        if (self.index == 0 and self.bot_pos.Y < self.ball_pos.Y) or (
                self.index == 1 and self.bot_pos.Y > self.ball_pos.Y):
            self.aim(self.ball_pos.X, self.ball_pos.Y)

            # self.check_for_dodge(self.ball_pos.X, self.ball_pos.Y, values)

        else:
            if self.index == 0:
                # Blue team's goal is located at (0, -5000)
                self.aim(0, -5000)
            else:
                # Orange team's goal is located at (0, 5000)
                self.aim(0, 5000)

        # Boost on kickoff
        if self.ball_pos.X == 0 and self.ball_pos.Y == 0:
            self.aim(self.ball_pos.X, self.ball_pos.Y)
            self.boost = True
            self.throttle = 1

        return [self.throttle, self.steer,
                self.pitch, self.yaw, self.roll,
                self.jump, self.boost, self.powerslide]
