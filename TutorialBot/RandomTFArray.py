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



    def get_game_info(self, batch_size):
        info = self.create_object()
        # Game info

        info.TimeSeconds = tf.constant([50.0] * batch_size)
        info.GameTimeRemaining = tf.random_uniform(shape=[batch_size, ], maxval=300, dtype=tf.float32)
        info.bOverTime = self.false
        info.bUnlimitedTime = self.false
        info.bRoundActive = self.true
        info.bBallHasBeenHit = self.true
        info.bMatchEnded = self.false

        return info

    def create_3D_point(self, x, y, z):
        point = self.create_object()
        point.X = x
        point.Y = y
        point.Z = z

        return point

    def create_3D_rotation(self, pitch, yaw, roll):
        rotator = self.create_object()
        rotator.Pitch = pitch
        rotator.Yaw = yaw
        rotator.Roll = roll

        return rotator

    def createRotVelAng(self, batch_size):
        rotation = self.create_3D_rotation(
            tf.random_uniform(shape=[batch_size, ], minval=-16384, maxval=16384, dtype=tf.float32), # Pitch
            tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=32768, dtype=tf.float32), # Yaw
            tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=32768, dtype=tf.float32)) # Roll

        velocity = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=2300, dtype=tf.float32), # Velocity X
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=2300, dtype=tf.float32), # Y
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=2300, dtype=tf.float32)) # Z

        angular = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=5.5, dtype=tf.float32), # Angular velocity X
            tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=5.5, dtype=tf.float32), # Y
            tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=5.5, dtype=tf.float32)) # Z

        return (rotation, velocity, angular)

    def createEmptyRotVelAng(self, batch_size):

        # Pitch -56 (basically 0) car at rest. 16384 nose of car facing up. -16384 car nose facing down. 0 when upside down.
        # Roll  0 at rest. -32768 bottom of wheels facing ceiling.
        # -16384 bottom of wheels facing side wall with right wheels higher than left wheels
        # 16384 bottom of wheels facing side wall with left wheels higher than right wheels (from 3rd person perspective behind car).

        rotation = self.create_3D_rotation(
                                           self.zero, # Pitch
                                           self.zero, # Yaw
                                           self.zero) # Roll

        velocity = self.create_3D_point(
                                        self.zero, # Velocity X
                                        self.zero, # Y
                                        self.zero) # Z

        angular = self.create_3D_point(
                                       self.zero, # Angular velocity X
                                       self.zero, # Y
                                       self.zero) # Z

        return (rotation, velocity, angular)

    def get_on_ground_info(self, is_on_ground):
        return tf.cond(is_on_ground,
                       lambda: tf.random_uniform(shape=[ ], maxval=16.7, dtype=tf.float32),  # Z on ground
                       lambda: tf.random_uniform(shape=[ ], minval=16.7, maxval=2000, dtype=tf.float32)) # Z in air

    def get_normal_car_values(self, batch_size, is_on_ground):
        car_z = tf.map_fn(self.get_on_ground_info, is_on_ground, dtype=tf.float32)
        location = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32), # X
            tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32), # Y
            car_z)

        rotation, velocity, angularVelocity = self.createRotVelAng(batch_size)

        return (location, rotation, velocity, angularVelocity)

    def get_car_info(self, batch_size, is_on_ground, team, index):
        car = self.create_object()

        get_normal_car_values = self.get_normal_car_values

        car.Location, car.Rotation, car.Velocity, car.AngularVelocity = tf.cond(
            tf.logical_and(team == 0, tf.greater_equal(tf.random_uniform(shape=[ ], maxval=0.6, dtype=tf.float32), 0.5)),
            lambda: get_normal_car_values(batch_size, is_on_ground),
            self.get_kickoff_data)


        car.bDemolished = self.false # Demolished

        car.bOnGround = is_on_ground

        car.bJumped = tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.6, dtype=tf.float32)) # Jumped

        car.bDoubleJumped = tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.55, dtype=tf.float32)) # Double jumped

        car.Team = team # Team

        car.Boost = tf.to_float(tf.random_uniform(shape=[batch_size, ], maxval=101, dtype=tf.int32)) # Boost

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

    def get_empty_car_info(self, batch_size, is_on_ground, team, index):
        car = self.create_object()
        car.Location = self.create_3D_point(
                                            self.zero, # X
                                            self.zero, # Y
                                            self.zero) # Z in air

        car.Rotation, car.Velocity, car.AngularVelocity = self.createEmptyRotVelAng(batch_size)

        car.bDemolished = self.false # Demolished

        car.bOnGround = is_on_ground

        car.bJumped = self.false # Jumped

        car.bDoubleJumped = self.false # Double jumped

        car.Team = team # Team

        car.Boost = self.zero # Boost

        car.Score = self.get_car_score_info()
        car.wName = index

        return car


    def get_ball_info(self, batch_size):
        ball = self.create_object()
        ball.Location = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-4050, maxval=8100,  dtype=tf.float32),  # Location X
            tf.random_uniform(shape=[batch_size, ], minval=-5900, maxval=11800, dtype=tf.float32),  # Y
            tf.random_uniform(shape=[batch_size, ], minval=0,     maxval=2000,  dtype=tf.float32))  # Z

        ball.Rotation, ball.Velocity, ball.AngularVelocity = self.createRotVelAng(batch_size)

        ball.Acceleration = self.create_3D_point(
                                                 self.zero, # Acceleration X
                                                 self.zero, # Acceleration Y
                                                 self.zero) # Acceleration Z

        ball.LatestTouch = self.create_object()
        ball.LatestTouch.wPlayerName = tf.round(tf.random_uniform(shape=[batch_size, ], minval=0, maxval=2,  dtype=tf.float32))

        ball.LatestTouch.sHitLocation = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-4050, maxval=8100,  dtype=tf.float32),  # Location X
            tf.random_uniform(shape=[batch_size, ], minval=-5900, maxval=11800, dtype=tf.float32),  # Y
            tf.random_uniform(shape=[batch_size, ], minval=0,     maxval=2000,  dtype=tf.float32))  # Z

        # INVALID VALUES
        ball.LatestTouch.sHitNormal = self.create_3D_point(
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32), # Velocity X
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32), # Y
            tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32)) # Z
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
            boost_info.Location = self.create_3D_point(tf.constant([boost_array[i * 5]] * batch_size),
                                                       tf.constant([boost_array[i * 5 + 1]] * batch_size),
                                                       tf.constant([boost_array[i * 5 + 2]] * batch_size))
            boost_info.bActive = tf.constant([boost_array[i * 5 + 3]] * batch_size)
            boost_info.Timer = tf.constant([boost_array[i * 5 + 4]] * batch_size)
            boost_objects.append(boost_info)
        return boost_objects

    def get_random_array(self, batch_size):
        is_on_ground = tf.greater(tf.random_uniform(shape=[batch_size, ], maxval=2, dtype=tf.int32), 1)

        game_tick_packet = self.create_object()
        # Game info
        game_tick_packet.gameInfo = self.get_game_info(batch_size)
        # Score info

        # Player car info
        game_tick_packet.gamecars = []
        game_tick_packet.gamecars.append(self.get_car_info(batch_size, is_on_ground, self.zero, self.zero))

        game_tick_packet.numCars = len(game_tick_packet.gamecars)

        # Ball info
        game_tick_packet.gameball = self.get_ball_info(batch_size)

        # Teammates info, 1v1 so empty
        self.get_empty_car_info(batch_size, is_on_ground, self.zero, self.two)
        self.get_empty_car_info(batch_size, is_on_ground, self.zero, self.four)

        # Enemy info, 1 enemy
        game_tick_packet.gamecars.append(self.get_car_info(batch_size, is_on_ground, self.one, self.one))
        self.get_empty_car_info(batch_size, is_on_ground, self.one, self.three)
        self.get_empty_car_info(batch_size, is_on_ground, self.one, self.five)

        game_tick_packet.gameBoosts = self.get_boost_info(batch_size)

        return game_tick_packet

    def get_kickoff_data(self):
        locations = [[-0.0009841920109465718, -4607.98583984375, 17.02585220336914], # center
         [255.99899291992188, -3839.99072265625, 17.02585220336914], # center left
         [-256.0019836425781, -3839.990966796875, 17.02585220336914], # center right
         [1951.9951171875, -2463.99169921875, 17.025854110717773], # diagonal left
         [-1952.0006103515625, -2463.991455078125, 17.02585220336914]] # diagonal right
        yaw = [16384.0, # center
               16384.0, # center left
               16384.0, # center right
               24576.0, # diagonal left
               8192.0]  # diagonal right

        locations = tf.constant(locations)
        yaw = tf.constant(yaw)

        # yaw +/- 20

        random_index = tf.random_uniform(shape=[batch_size, ], minval=0, maxval=len(yaw), dtype=tf.int8),

        kick_off_loc, yaw = self.slice_kickoff_locations(random_index, locations, yaw)
        kick_off_loc.set_shape([batch_size, 3])
        kick_off_loc = tf.reshape([3, batch_size])
        yaw.set_shape([batch_size, ])
        location = self.create_3D_point(kick_off_loc[0],
                                        kick_off_loc[1],
                                        kick_off_loc[2])

        yaw = yaw + tf.random_uniform(shape=[batch_size, ], minval=0, maxval=20, dtype=tf.int8)

        rotation = self.create_3D_rotation(tf.constant(0),
                                           yaw,
                                           tf.constant(0))

        _, velocity, angular = self.createEmptyRotVelAng(self.batch_size)

        return (location, rotation, velocity, angular)

    def slice_kickoff_locations(self, random_index, locations, yaw):

        def get_kickoff_location(elements):
            random_index = elements[0]
            return (tf.slice(locations, [random_index], [1]),
                    tf.slice(yaw, [random_index], [1])),

        return tf.map_fn(
            get_kickoff_location
            (random_index, random_index),
            dtype=tf.float32,
            infer_shape = False)
