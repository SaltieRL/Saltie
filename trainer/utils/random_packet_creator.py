import tensorflow as tf


class TensorflowPacketGenerator:
    def __init__(self, batch_size):
        self.zero = tf.constant([0.0] * batch_size)
        self.false = self.zero
        self.one = tf.constant([1.0] * batch_size)
        self.true = self.one
        self.two = tf.constant([2.0] * batch_size)
        self.three = tf.constant([3.0] * batch_size)
        self.four = tf.constant([4.0] * batch_size)
        self.five = tf.constant([5.0] * batch_size)
        self.batch_size = batch_size

    # game_info + score_info + player_car + ball_data +
    # self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info
    def create_object(self):
        return lambda: None

    def get_game_info(self):
        info = self.create_object()
        # Game info
        batch_size = self.batch_size
        info.TimeSeconds = tf.constant([50.0] * batch_size)
        info.GameTimeRemaining = tf.random_uniform(shape=[batch_size, ], maxval=300, dtype=tf.float32)
        info.bOverTime = self.false
        info.bUnlimitedTime = self.false
        info.bRoundActive = self.true
        info.bBallHasBeenHit = self.true
        info.bMatchEnded = self.false

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

    def remove_number(self, input, not_on_ground):
        scale_amount = tf.constant(1000.0, dtype=tf.float32)
        input_scaled = tf.cast(input * scale_amount, tf.int32)
        post_bitwise = tf.bitwise.bitwise_and(input_scaled, not_on_ground)
        output = tf.cast(post_bitwise, tf.float32) / scale_amount
        return output

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

    def create_mask(self, value):
        value = tf.cast(tf.equal(value, False), tf.int32)
        added_value = tf.constant(0)
        two = tf.constant(2, dtype=tf.int32)
        for i in range(32):
            added_value += value * tf.pow(two, tf.constant(i, dtype=tf.int32))
        return added_value

    def createRotVelAng(self, is_on_ground):
        batch_size = self.batch_size
        with tf.name_scope("Rotation"):
            rotation = self.create_3D_rotation(
                tf.random_uniform(shape=[batch_size, ], minval=-16384, maxval=16384, dtype=tf.float32),  # Pitch
                tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=32768, dtype=tf.float32),  # Yaw
                tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=32768, dtype=tf.float32))  # Roll

        with tf.name_scope("Velocity"):
            vel_rand = tf.random_uniform(shape=[3, batch_size, ], minval=-2300, maxval=2300, dtype=tf.float32)
            velocity = self.create_3D_point(vel_rand[0],  # Velocity X
                                            vel_rand[1],  # Y
                                            vel_rand[2])  # Z

        with tf.name_scope("AngularVelocity"):
            ang_rand = tf.random_uniform(shape=[3, batch_size, ], minval=-5.5, maxval=5.5, dtype=tf.float32)
            angular = self.create_3D_point(ang_rand[0], # Angular velocity X
                                           ang_rand[1],  # Y
                                           ang_rand[2])  # Z

        #reverse ground data
        mask = self.create_mask(is_on_ground)

        rotation.Pitch = self.remove_number(rotation.Pitch, mask)
        rotation.Roll = self.remove_number(rotation.Roll, mask)
        velocity.Z = self.remove_number(velocity.Z, mask)
        angular.Z = self.remove_number(angular.Z, mask)

        return (rotation, velocity, angular)

    def createEmptyRotVelAng(self):

        # Pitch -56 (basically 0) car at rest. 16384 nose of car facing up. -16384 car nose facing down. 0 when upside down.
        # Roll  0 at rest. -32768 bottom of wheels facing ceiling.
        # -16384 bottom of wheels facing side wall with right wheels higher than left wheels
        # 16384 bottom of wheels facing side wall with left wheels higher than right wheels (from 3rd person perspective behind car).

        with tf.name_scope("Rotation"):
            rotation = self.create_3D_rotation(
                self.zero,  # Pitch
                self.zero,  # Yaw
                self.zero)  # Roll

        with tf.name_scope("Velocity"):
            velocity = self.create_3D_point(
                self.zero,  # Velocity X
                self.zero,  # Y
                self.zero)  # Z

        with tf.name_scope("AngularVelocity"):
            angular = self.create_3D_point(
                self.zero,  # Angular velocity X
                self.zero,  # Y
                self.zero)  # Z

        return (rotation, velocity, angular)

    def get_on_ground_info(self, is_on_ground):
        return tf.cond(is_on_ground,
                       lambda: tf.random_uniform(shape=[], minval=0.0, maxval=16.7, dtype=tf.float32),  # Z on ground
                       lambda: tf.random_uniform(shape=[], minval=16.7, maxval=2000, dtype=tf.float32))  # Z in air

    def get_normal_car_values(self, is_on_ground):
        batch_size = self.batch_size
        car_z = tf.map_fn(self.get_on_ground_info, is_on_ground, dtype=tf.float32)
        location = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-8100, maxval=8100, dtype=tf.float32),  # X
            tf.random_uniform(shape=[batch_size, ], minval=-11800, maxval=11800, dtype=tf.float32),  # Y
            car_z)

        rotation, velocity, angularVelocity = self.createRotVelAng(is_on_ground)

        return self.decompose(location, rotation, velocity, angularVelocity)

    def get_car_info(self, batch_size, is_on_ground, team, index, is_kickoff):
        car = self.create_object()

        get_normal_car_values = self.get_normal_car_values

        car.Location, car.Rotation, car.Velocity, car.AngularVelocity = self.convert_to_objects(tf.cond(
            tf.logical_and(team == 0, is_kickoff[0]),
            self.get_kickoff_data,
            lambda: get_normal_car_values(is_on_ground)))


        car.bDemolished = self.false  # Demolished

        car.bOnGround = is_on_ground

        car.bJumped = tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.6, dtype=tf.float32))  # Jumped
        car.bSuperSonic = self.false # Jumped

        car.bDoubleJumped = tf.round(
            tf.random_uniform(shape=[batch_size, ], maxval=0.55, dtype=tf.float32))  # Double jumped

        car.Team = team  # Team

        car.Boost = tf.to_float(tf.random_uniform(shape=[batch_size, ], maxval=101, dtype=tf.int32))  # Boost

        car.Score = self.get_car_score_info()

        car.wName = index

        return car

    def get_car_score_info(self):
        score = self.create_object()
        score.Score = self.zero
        score.Goals = self.zero
        score.OwnGoals = self.zero
        score.Assists = self.zero
        score.Saves = self.zero
        score.Shots = self.zero
        score.Demolitions = self.zero
        return score

    def get_empty_car_info(self, is_on_ground, team, index):
        car = self.create_object()
        car.Location = self.create_3D_point(
            self.zero,  # X
            self.zero,  # Y
            self.zero)  # Z in air

        car.Rotation, car.Velocity, car.AngularVelocity = self.createEmptyRotVelAng()

        car.bDemolished = self.false  # Demolished

        car.bOnGround = is_on_ground

        car.bSuperSonic = self.false # Jumped

        car.bJumped = self.false  # Jumped

        car.bDoubleJumped = self.false  # Double jumped

        car.Team = team  # Team

        car.Boost = self.zero  # Boost

        car.Score = self.get_car_score_info()
        car.wName = index

        return car

    def get_ball_info(self, batch_size, is_on_ground, is_kickoff):
        ball = self.create_object()
        ball.Location = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-8100, maxval=8100, dtype=tf.float32),  # Location X
            tf.random_uniform(shape=[batch_size, ], minval=-11800, maxval=11800, dtype=tf.float32),  # Y
            tf.random_uniform(shape=[batch_size, ], minval=0, maxval=2000, dtype=tf.float32))  # Z

        kick_off_mask = self.create_mask(is_kickoff)
        on_ground_mask = self.create_mask(is_on_ground)

        merged_mask = tf.bitwise.bitwise_and(kick_off_mask, on_ground_mask)

        ball.Rotation, ball.Velocity, ball.AngularVelocity = self.createRotVelAng(is_on_ground)

        with tf.name_scope("BallAccerlation"):
            ball.Acceleration = self.create_3D_point(
                self.zero,  # Acceleration X
                self.zero,  # Acceleration Y
                self.zero)  # Acceleration Z

        ball.LatestTouch = self.create_object()

        ball.LatestTouch.wPlayerName = tf.round(tf.random_uniform(shape=[batch_size, ],
                                                                  minval=0, maxval=2, dtype=tf.float32),
                                                name="wPlayerName")

        with tf.name_scope("HitLocation"):
            ball.LatestTouch.sHitLocation = self.create_3D_point(
                tf.random_uniform(shape=[batch_size, ], minval=-8100, maxval=8100, dtype=tf.float32),  # Location X
                tf.random_uniform(shape=[batch_size, ], minval=-11800, maxval=11800, dtype=tf.float32),  # Y
                tf.random_uniform(shape=[batch_size, ], minval=0, maxval=2000, dtype=tf.float32), )  # Z
        with tf.name_scope("HitNormal"):
            ball.LatestTouch.sHitNormal = self.create_3D_point(
                tf.random_uniform(shape=[batch_size, ], minval=-6000, maxval=6000, dtype=tf.float32),  # Velocity X
                tf.random_uniform(shape=[batch_size, ], minval=-6000, maxval=6000, dtype=tf.float32),  # Y
                tf.random_uniform(shape=[batch_size, ], minval=-6000, maxval=6000, dtype=tf.float32))  # Z

        ball.Location.Z = self.remove_number(ball.Velocity.Z, on_ground_mask)

        ball.Location.X = self.remove_number(ball.Location.X, kick_off_mask)
        ball.Location.Y = self.remove_number(ball.Location.Y, kick_off_mask)
        ball.Velocity.X = self.remove_number(ball.Velocity.X, kick_off_mask)
        ball.Velocity.Y = self.remove_number(ball.Velocity.Y, kick_off_mask)

        ball.Velocity.Z = self.remove_number(ball.Velocity.Z, merged_mask)
        return ball

    def get_boost_info(self, batch_size):
        boost_objects = []
        boost_array = [2048.0, -1036.0, 64.0, 1.0, 4000, -1772.0, -2286.247802734375, 64.0, 1.0, 4000, 0.0, -2816.0,
                       64.0, 1.0, 4000, -2048.0, -1036.0, 64.0, 1.0, 4000, -3584.0, -2484.0, 64.0, 1.0, 4000, 1772.0,
                       -2286.247802734375, 64.0, 1.0, 4000, 3328.0009765625, 4096.0, 136.0, 1.0, 0,
                       -3071.999755859375, 4096.0, 72.00000762939453, 1.0, 10000, 3072.0, -4095.99951171875,
                       72.00000762939453, 1.0, 10000, -3072.0, -4095.9990234375, 72.00000762939453, 1.0, 10000,
                       -3584.0, 1.1190114491910208e-05, 72.00000762939453, 1.0, 10000, 3584.0, 0.0, 72.00000762939453,
                       1.0, 10000, 3071.9921875, 4096.0, 72.00000762939453, 1.0, 10000, -1792.0, -4184.0, 64.0, 1.0,
                       4000, 1792.0, -4184.0, 64.0, 1.0, 4000, -940.0, -3308.0, 64.0, 1.0, 4000, 940.0, -3308.0, 64.0,
                       1.0, 4000, 3584.0, -2484.0, 64.0, 1.0, 4000, 0.0, 1024.0, 64.0, 1.0, 4000, -2048.0, 1036.0,
                       64.0, 1.0, 4000, -1772.0, 2284.0, 64.0, 1.0, 4000, 2048.0, 1036.0, 64.0, 1.0, 4000, 1772.0,
                       2284.0, 64.0, 1.0, 4000, 3584.0, 2484.0, 64.0, 1.0, 4000, 1792.0, 4184.0, 64.0, 1.0, 4000,
                       -1792.0, 4184.0, 64.0, 1.0, 4000, 0.0, 2816.0, 64.0, 1.0, 4000, -939.9991455078125,
                       3307.99951171875, 64.0, 1.0, 4000, -3584.0, 2484.0, 64.0, 1.0, 4000, 940.0, 3308.0, 64.0, 1.0,
                       4000, 0.0, 4240.0, 64.0, 1.0, 4000, 1024.0, 0.0, 64.0, 1.0, 4000, 0.0, -1024.0, 64.0, 1.0,
                       4000, -1024.0, 0.0, 64.0, 1.0, 4000, 0.0, -4240.0, 64.0, 1.0, 4000]
        for i in range(35):
            boost_info = self.create_object()
            with tf.name_scope('BoostLocation'):
                boost_info.Location = self.create_3D_point(tf.constant([boost_array[i * 5]] * batch_size),
                                                           tf.constant([boost_array[i * 5 + 1]] * batch_size),
                                                           tf.constant([boost_array[i * 5 + 2]] * batch_size))
            boost_info.bActive = tf.constant([boost_array[i * 5 + 3]] * batch_size, name='BostActive')
            boost_info.Timer = tf.constant([boost_array[i * 5 + 4]] * batch_size, name='BoostTimer')
            boost_objects.append(boost_info)
        return boost_objects

    def get_random_array(self):
        batch_size = self.batch_size
        is_car_on_ground = tf.greater_equal(tf.random_uniform(shape=[batch_size, ], minval=0, maxval=2, dtype=tf.int32), 1)
        is_ball_on_ground = tf.greater_equal(tf.random_uniform(shape=[batch_size, ], minval=0, maxval=2, dtype=tf.int32), 1)
        is_kickoff = tf.greater_equal(tf.random_uniform(shape=[batch_size, ], minval=0, maxval=10, dtype=tf.int32), 8)

        game_tick_packet = self.create_object()
        # Game info
        with tf.name_scope("Game_Info"):
            game_tick_packet.gameInfo = self.get_game_info()
        # Score info

        # Player car info
        game_tick_packet.gamecars = []
        with tf.name_scope("Player_Car"):
            game_tick_packet.gamecars.append(self.get_car_info(batch_size, is_car_on_ground, self.zero, self.zero,
                                                               is_kickoff))

        game_tick_packet.numCars = len(game_tick_packet.gamecars)

        # Ball info
        with tf.name_scope("Ball_Info"):
            game_tick_packet.gameball = self.get_ball_info(batch_size, is_ball_on_ground, is_kickoff)

        # Teammates info, 1v1 so empty
        with tf.name_scope("Team_0"):
            self.get_empty_car_info(is_car_on_ground, self.zero, self.two)
            self.get_empty_car_info(is_car_on_ground, self.zero, self.four)

        # Enemy info, 1 enemy
        with tf.name_scope("Enemy"):
            game_tick_packet.gamecars.append(self.get_car_info(batch_size, is_car_on_ground, self.one, self.one,
                                                               is_kickoff))
        with tf.name_scope("Enemy1"):
            self.get_empty_car_info(is_car_on_ground, self.one, self.three)
            self.get_empty_car_info(is_car_on_ground, self.one, self.five)

        with tf.name_scope("Boost"):
            game_tick_packet.gameBoosts = self.get_boost_info(batch_size)

        return game_tick_packet

    def get_kickoff_data(self):
        locations = [[-0.0009841920109465718, -4607.98583984375, 17.02585220336914],  # center
                     [255.99899291992188, -3839.99072265625, 17.02585220336914],  # center left
                     [-256.0019836425781, -3839.990966796875, 17.02585220336914],  # center right
                     [1951.9951171875, -2463.99169921875, 17.025854110717773],  # diagonal left
                     [-1952.0006103515625, -2463.991455078125, 17.02585220336914]]  # diagonal right
        yaw_list = [16384.0,  # center
                    16384.0,  # center left
                    16384.0,  # center right
                    24576.0,  # diagonal left
                    8192.0]  # diagonal right

        length_of_positions = len(yaw_list)

        locations = tf.constant(locations)

        yaw_list = tf.constant(yaw_list)

        # yaw +/- 20

        random_index = tf.round(
            tf.random_uniform(shape=[self.batch_size, ], minval=0, maxval=length_of_positions - 1, dtype=tf.float32))
        random_index = tf.cast(random_index, tf.int32)

        kick_off_loc, yaw = self.slice_kickoff_locations(random_index, locations, yaw_list)
        kick_off_loc.set_shape([self.batch_size, 3])
        kick_off_loc = tf.reshape(kick_off_loc, [3, self.batch_size])
        yaw.set_shape([self.batch_size, ])

        with tf.name_scope('Location'):
            location = self.create_3D_point(kick_off_loc[0],
                                            kick_off_loc[1],
                                            kick_off_loc[2])

        yaw = yaw + tf.random_uniform(shape=[self.batch_size, ], minval=0, maxval=20, dtype=tf.float32)

        with tf.name_scope('Rotation'):
            rotation = self.create_3D_rotation(self.zero,
                                               yaw,
                                               self.zero)

        _, velocity, angular = self.createEmptyRotVelAng()

        return self.decompose(location, rotation, velocity, angular)

    def slice_kickoff_locations(self, random_index, location_list, yaw_list):
        def get_kickoff_location(random_index):
            return (tf.slice(location_list, [random_index, 0], [1, -1]),
                    tf.slice(yaw_list, [random_index], [1]))

        result = tf.map_fn(
            get_kickoff_location,
            random_index,
            dtype=(tf.float32, tf.float32),
            infer_shape=False)

        location, yaw = result

        yaw = tf.squeeze(yaw)

        return (location, yaw)

    def decompose(self, *args):
        array = []
        for object in args:
            for item in object.__dir__():
                if not item.startswith('__'):
                    array.append(getattr(object, item))
        return array

    def convert_to_objects(self, array):
        return (
            self.create_3D_point(array[0],
                                 array[1],
                                 array[2],
                                 True),
            self.create_3D_rotation(array[3],
                                    array[4],
                                    array[5],
                                    True),
            self.create_3D_point(array[6],
                                 array[7],
                                 array[8],
                                 True),
            self.create_3D_point(array[9],
                                 array[10],
                                 array[11],
                                 True),
        )
