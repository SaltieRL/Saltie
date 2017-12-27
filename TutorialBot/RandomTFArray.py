import tensorflow as tf


# game_info + score_info + player_car + ball_data +
# self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info

def get_random_array(batch_size):
    array = []
    # Game info
    array.append(tf.constant([50.0] * batch_size))  # TimeSeconds
    array.append(tf.random_uniform(shape=[batch_size, ], maxval=300, dtype=tf.float32))  # Remaining time
    array.append(tf.constant([0.0] * batch_size))  # Overtime
    array.append(tf.constant([0.0] * batch_size))  # Unlimited time
    array.append(tf.constant([1.0] * batch_size))  # Round active
    array.append(tf.constant([1.0] * batch_size))  # Ball hit
    array.append(tf.constant([0.0] * batch_size))  # Match ended

    # Score info

    # Player car info
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32))  # Location X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32))  # Y
    array.append(tf.cond(tf.equal(tf.random_uniform(shape=[], maxval=2, dtype=tf.int32), 1),
                         lambda: tf.random_uniform(shape=[batch_size, ], maxval=16.7, dtype=tf.float32),  # Z on ground
                         lambda: tf.random_uniform(shape=[batch_size, ], minval=16.7, maxval=2000,
                                                   dtype=tf.float32)))  # Z off ground
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-16384, maxval=32768, dtype=tf.float32))  # Rotation Pitch
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Yaw
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Roll
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Z
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Angular velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Z
    array.append(tf.constant([0.0] * batch_size))  # Demolished
    array.append(tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.6, dtype=tf.float32)))  # Jumped
    array.append(tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.55, dtype=tf.float32)))  # Double jumped
    array.append(tf.constant([0.0] * batch_size))  # Team
    array.append(tf.to_float(tf.random_uniform(shape=[batch_size, ], maxval=101, dtype=tf.int32)))  # Boost

    # Ball info
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-4050, maxval=8100, dtype=tf.float32))  # Location X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5900, maxval=11800, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], maxval=2000, dtype=tf.float32))  # Z
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-16384, maxval=32768, dtype=tf.float32))  # Rotation Pitch
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Yaw
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Roll
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Z
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Angular velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Z
    array.append(tf.constant([0.0] * batch_size))  # Acceleration X
    array.append(tf.constant([0.0] * batch_size))  # Y
    array.append(tf.constant([0.0] * batch_size))  # Z

    # Teammates info, 1v1 so empty

    # Enemy info, 1 enemy
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32))  # Location X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-3800, maxval=7600, dtype=tf.float32))  # Y
    array.append(tf.cond(tf.equal(tf.random_uniform(shape=[], maxval=2, dtype=tf.int32), 1),
                         lambda: tf.random_uniform(shape=[batch_size, ], maxval=16.7, dtype=tf.float32),  # Z on ground
                         lambda: tf.random_uniform(shape=[batch_size, ], minval=16.7, maxval=2000,
                                                   dtype=tf.float32)))  # Z off ground
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-16384, maxval=32768, dtype=tf.float32))  # Rotation Pitch
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Yaw
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-32768, maxval=65536, dtype=tf.float32))  # Roll
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-2300, maxval=4600, dtype=tf.float32))  # Z
    array.append(
        tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Angular velocity X
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Y
    array.append(tf.random_uniform(shape=[batch_size, ], minval=-5.5, maxval=11, dtype=tf.float32))  # Z
    array.append(tf.constant([0.0] * batch_size))  # Demolished
    array.append(tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.6, dtype=tf.float32)))  # Jumped
    array.append(tf.round(tf.random_uniform(shape=[batch_size, ], maxval=0.55, dtype=tf.float32)))  # Double jumped
    array.append(tf.constant([1.0] * batch_size))  # Team
    array.append(tf.to_float(tf.random_uniform(shape=[batch_size, ], maxval=101, dtype=tf.int32)))  # Boost

    # Boost info
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([0.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([10000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    array.append(tf.constant([1.0] * batch_size))
    array.append(tf.constant([4000.0] * batch_size))
    return tf.stack(array)
