import tensorflow as tf
from conversions.input.normalization_input_formatter import NormalizationInputFormatter


class DataNormalizer:
    formatter = NormalizationInputFormatter(0, 0, None)
    boolean = [0.0, 1.0]

    # game_info + score_info + player_car + ball_data +
    # self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info
    def create_object(self):
        return lambda: None

    def get_game_info(self):
        info = self.create_object()
        # Game info
        info.bOverTime = self.boolean
        info.bUnlimitedTime = self.boolean
        info.bRoundActive = self.boolean
        info.bBallHasBeenHit = self.boolean
        info.bMatchEnded = self.boolean

        return info

    def create_3D_point(self, x, y, z, convert_name=True):
        point = self.create_object()
        if convert_name:
            point.X = tf.identity(x, name='X')
            point.Y = tf.identity(y, name='Y')
            point.Z = tf.identity(z, name='Z')
        else:
            point.X = x
            point.Y = y
            point.Z = z

        return point

    def create_3D_rotation(self, pitch, yaw, roll, convert_name=True):
        rotator = self.create_object()
        if convert_name:
            rotator.Pitch = tf.identity(pitch, name='Pitch')
            rotator.Yaw = tf.identity(yaw, name='Yaw')
            rotator.Roll = tf.identity(roll, name='Roll')
        else:
            rotator.Pitch = pitch
            rotator.Yaw = yaw
            rotator.Roll = roll
        return rotator

    def createRotVelAng(self, input_velocity, input_angular):
        with tf.name_scope("Rotation"):
            rotation = self.create_3D_rotation([-16384, 16384],  # Pitch
                                               [-32768, 32768],  # Yaw
                                               [-32768, 32768])  # Roll

        with tf.name_scope("Velocity"):
            velocity = self.create_3D_point(
                [-input_velocity, input_velocity],  # Velocity X
                [-input_velocity, input_velocity],  # Y
                [-input_velocity, input_velocity])  # Z

        with tf.name_scope("AngularVelocity"):
            angular = self.create_3D_point(
                [-input_angular, input_angular],  # Angular velocity X
                [-input_angular, input_angular],  # Y
                [-input_angular, input_angular])  # Z

        return (rotation, velocity, angular)

    def get_location(self):
        return self.create_3D_point(
            [-8300, 8300],  # Location X
            [-11800, 11800],  # Y
            [0, 2000])

    def get_car_info(self):
        car = self.create_object()

        car.Location = self.get_location()

        car.Rotation, car.Velocity, car.AngularVelocity = self.createRotVelAng(2300, 5.5)

        car.bDemolished = self.boolean  # Demolished

        car.bOnGround = self.boolean

        car.bJumped = self.boolean  # Jumped
        car.bSuperSonic = self.boolean # Jumped

        car.bDoubleJumped = self.boolean

        car.Team = self.boolean

        car.Boost = [0.0, 100]

        car.Score = self.get_car_score_info()

        return car

    def get_car_score_info(self):
        score = self.create_object()
        score.Score = [0, 100]
        score.Goals = self.boolean
        score.OwnGoals = self.boolean
        score.Assists = self.boolean
        score.Saves = self.boolean
        score.Shots = self.boolean
        score.Demolitions = self.boolean
        return score

    def get_ball_info(self):
        ball = self.create_object()
        ball.Location = self.create_3D_point(
            [-8300, 8300],  # Location X
            [-11800, 11800],  # Y
            [0, 2000])  # Z

        ball.Rotation, ball.Velocity, ball.AngularVelocity = self.createRotVelAng(6000.0, 6.0)

        with tf.name_scope("BallAccerlation"):
            ball.Acceleration = self.create_3D_point(
                self.boolean,  # Acceleration X
                self.boolean,  # Acceleration Y
                self.boolean)  # Acceleration Z

        ball.LatestTouch = self.create_object()

        with tf.name_scope("HitLocation"):
            ball.LatestTouch.sHitLocation = self.get_location()
        with tf.name_scope("HitNormal"):
            ball.LatestTouch.sHitNormal = ball.Velocity
        return ball

    def get_boost_info(self):
        boost_objects = []
        for i in range(35):
            boost_info = self.create_object()
            with tf.name_scope('BoostLocation'):
                boost_info.Location = self.get_location()
            boost_info.bActive = self.boolean
            boost_info.Timer = [0.0, 10000.0]
            boost_objects.append(boost_info)
        return boost_objects

    def get_normalization_array(self):
        game_tick_packet = self.create_object()
        # Game info
        with tf.name_scope("Game_Info"):
            game_tick_packet.gameInfo = self.get_game_info()
        # Score info

        # Player car info
        game_tick_packet.gamecars = []
        car_info = self.get_car_info()
        for i in range(6):
            game_tick_packet.gamecars.append(car_info)

        game_tick_packet.numCars = len(game_tick_packet.gamecars)

        # Ball info
        with tf.name_scope("Ball_Info"):
            game_tick_packet.gameball = self.get_ball_info()

        with tf.name_scope("Boost"):
            game_tick_packet.gameBoosts = self.get_boost_info()
        return self.formatter.create_input_array(game_tick_packet)[0]
