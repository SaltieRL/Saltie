import random as r


class PacketGenerator:
    def __init__(self):
        self.boosts = self.get_boost_info()

    def get_random_packet(self):
        game_tick_packet = self.create_object()
        game_tick_packet.gamecars = self.get_car_info(2)
        game_tick_packet.numCars = 2
        game_tick_packet.gameBoosts = self.boosts
        game_tick_packet.numBoosts = 35
        game_tick_packet.gameball = self.get_ball_info()
        game_tick_packet.gameInfo = self.get_game_info()
        return game_tick_packet

    def get_car_info(self, num_cars):
        car_objects = []
        for i in range(num_cars):
            if r.randrange(0, 1) == 0:
                car_objects.append(self.get_car_on_ground(i))
            else:
                car_objects.append(self.get_car_off_ground(i))
        return car_objects

    def get_car_on_ground(self, index):
        car = self.create_object()
        car.Location = self.create_3D_point(-3800 + 7600 * r.random(),
                                            -5600 + 11200 * r.random(),
                                            20 * r.random())
        car.Rotation = self.create_3D_rotation(-16384 + 32768 * r.random(),
                                               -32768 + 65536 * r.random(),
                                               -32768 + 65536 * r.random())
        car.Velocity = self.create_3D_point(-2300 + 4600 * r.random(),
                                            -2300 + 4600 * r.random(),
                                            -2300 + 4600 * r.random())
        car.AngularVelocity = self.create_3D_point(-5.5 + 11 * r.random(),
                                                   -5.5 + 11 * r.random(),
                                                   -5.5 + 11 * r.random())
        car.Score = self.get_car_score_info()
        car.bDemolished = False
        car.bOnGround = True
        car.bSuperSonic = False  # Should be a precise method through velocities
        car.bBot = True
        car.bJumped = bool(round(0.6 * r.random()))
        car.bDoubleJumped = bool(round(0.55 * r.random()))
        car.wName = "RandomizedBotData" + str(index)
        car.Team = index
        car.Boost = r.randint(0, 100)
        return car

    def get_car_off_ground(self, index):
        car = self.create_object()
        car.Location = self.create_3D_point(-3800 + 7600 * r.random(),
                                            -5600 + 11200 * r.random(),
                                            20 + 1980 * r.random())
        car.Rotation = self.create_3D_rotation(-16384 + 32768 * r.random(),
                                               -32768 + 65536 * r.random(),
                                               -32768 + 65536 * r.random())
        car.Velocity = self.create_3D_point(-2300 + 4600 * r.random(),
                                            -2300 + 4600 * r.random(),
                                            -2300 + 4600 * r.random())
        car.AngularVelocity = self.create_3D_point(-5.5 + 11 * r.random(),
                                                   -5.5 + 11 * r.random(),
                                                   -5.5 + 11 * r.random())
        car.Score = self.get_car_score_info()
        car.bDemolished = False
        car.bOnGround = False  # Should be more precise
        car.bSuperSonic = False  # Should be a precise method through velocities
        car.bBot = True
        car.bJumped = bool(round(0.6 * r.random()))
        car.bDoubleJumped = bool(round(0.55 * r.random()))
        car.wName = "RandomizedBotData" + str(i)
        car.Team = index
        car.Boost = r.randint(0, 100)
        return car



    def get_car_score_info(self):
        score = self.create_object()
        score.Score = 230
        score.Goals = 4
        score.OwnGoals = 1
        score.Assists = 2
        score.Saves = 3
        score.Shots = 6
        score.Demolitions = 1
        return score

    def get_boost_info(self):
        boost_objects = []
        boost_array = [2048.0, -1036.0, 64.0, True, 4000, -1772.0, -2286.247802734375, 64.0, True, 4000, 0.0, -2816.0,
                       64.0, True, 4000, -2048.0, -1036.0, 64.0, True, 4000, -3584.0, -2484.0, 64.0, True, 4000, 1772.0,
                       -2286.247802734375, 64.0, True, 4000, 3328.0009765625, 4096.0, 136.0, True, 0,
                       -3071.999755859375, 4096.0, 72.00000762939453, True, 10000, 3072.0, -4095.99951171875,
                       72.00000762939453, True, 10000, -3072.0, -4095.9990234375, 72.00000762939453, True, 10000,
                       -3584.0, 1.1190114491910208e-05, 72.00000762939453, True, 10000, 3584.0, 0.0, 72.00000762939453,
                       True, 10000, 3071.9921875, 4096.0, 72.00000762939453, True, 10000, -1792.0, -4184.0, 64.0, True,
                       4000, 1792.0, -4184.0, 64.0, True, 4000, -940.0, -3308.0, 64.0, True, 4000, 940.0, -3308.0, 64.0,
                       True, 4000, 3584.0, -2484.0, 64.0, True, 4000, 0.0, 1024.0, 64.0, True, 4000, -2048.0, 1036.0,
                       64.0, True, 4000, -1772.0, 2284.0, 64.0, True, 4000, 2048.0, 1036.0, 64.0, True, 4000, 1772.0,
                       2284.0, 64.0, True, 4000, 3584.0, 2484.0, 64.0, True, 4000, 1792.0, 4184.0, 64.0, True, 4000,
                       -1792.0, 4184.0, 64.0, True, 4000, 0.0, 2816.0, 64.0, True, 4000, -939.9991455078125,
                       3307.99951171875, 64.0, True, 4000, -3584.0, 2484.0, 64.0, True, 4000, 940.0, 3308.0, 64.0, True,
                       4000, 0.0, 4240.0, 64.0, True, 4000, 1024.0, 0.0, 64.0, True, 4000, 0.0, -1024.0, 64.0, True,
                       4000, -1024.0, 0.0, 64.0, True, 4000, 0.0, -4240.0, 64.0, True, 4000]
        for i in range(35):
            boost_info = self.create_object()
            boost_info.Location = self.create_3D_point(boost_array[i * 5],
                                                       boost_array[i * 5 + 1],
                                                       boost_array[i * 5 + 2])
            boost_info.bActive = boost_array[i * 5 + 3]
            boost_info.Timer = boost_array[i * 5 + 4]
            boost_objects.append(boost_info)
        return boost_objects

    def get_ball_info(self):
        ball = self.create_object()
        ball.Location = self.create_3D_point(-4050 + 8100 * r.random(),
                                             -5900 + 11800 * r.random(),
                                             2000 * r.random())
        ball.Rotation = self.create_3D_rotation(-16384 + 32768 * r.random(),
                                                -32768 + 65536 * r.random(),
                                                -32768 + 65536 * r.random())
        ball.Velocity = self.create_3D_point(-2300 + 4600 * r.random(),
                                             -2300 + 4600 * r.random(),
                                             -2300 + 4600 * r.random())
        ball.AngularVelocity = self.create_3D_point(-5.5 + 11 * r.random(),
                                                    -5.5 + 11 * r.random(),
                                                    -5.5 + 11 * r.random())
        ball.Acceleration = self.create_3D_point(0,
                                                 0,
                                                 0)
        return ball

    def get_game_info(self):
        info = self.create_object()
        info.TimeSeconds = 50
        info.GameTimeRemaining = 300 * r.random()
        info.bOverTime = False
        info.bUnlimitedTime = False
        info.bRoundActive = True
        info.bBallHasBeenHit = True
        info.bMatchEnded = False
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

    def create_object(self):
        return lambda: None
