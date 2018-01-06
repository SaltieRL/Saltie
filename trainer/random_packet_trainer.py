import tensorflow as tf
import time
from conversions.input import input_formatter, tensorflow_input_formatter
from TutorialBot import tutorial_bot_output
from trainer.utils import random_packet_creator as r
from models.actor_critic import tutorial_model
from modelHelpers import action_handler
from trainer.utils import controller_statistics
from tqdm import tqdm


def get_random_data(packet_generator, input_formatter):
    game_tick_packet = packet_generator.get_random_array()
    output_array = input_formatter.create_input_array(game_tick_packet)[0]
    # reverse the shape of the array
    return output_array, game_tick_packet


learning_rate = 0.3
total_batches = 5000
batch_size = 2000
display_step = 1

# Network Parameters
n_neurons_hidden = 128  # every layer of neurons
n_input = input_formatter.get_state_dim_with_features()  # data input
n_output = 39  # total outputs

def calculate_loss(self, elements):
    throttle = elements[0]
    is_on_ground = elements[1]
    given_output = elements[2]
    created_output = elements[3]
    steer, powerslide, pitch, jump, boost = created_output

    def output_on_ground():
        # Throttle
        output = tf.losses.absolute_difference(throttle, given_output[0])

        # Steer
        # output += tf.cond(tf.less_equal(tf.abs(steer - given_output[1]), 0.5), lambda: 1, lambda: -1)
        output += tf.losses.absolute_difference(steer, given_output[1])

        # Powerslide
        # output += tf.cond(tf.equal(tf.cast(powerslide, tf.float32), given_output[7]), lambda: 1, lambda: -1)
        output += tf.losses.mean_squared_error(powerslide, given_output[1])
        return output

    def output_off_ground():
        # Pitch
        output = tf.losses.absolute_difference(pitch, given_output[2])
        # output = tf.cond(tf.less_equal(tf.abs(pitch - given_output[2]), 0.5), lambda: 1, lambda: -1)
        return output

    output = tf.cond(is_on_ground, output_on_ground, output_off_ground)

    # Jump
    # output += tf.cond(tf.equal(tf.cast(jump, tf.float32), given_output[5]), lambda: 1, lambda: -1)
    output += tf.losses.mean_squared_error(jump, given_output[5])

    # Boost
    # output += tf.cond(tf.equal(tf.cast(boost, tf.float32), given_output[6]), lambda: 1, lambda: -1)
    output += tf.losses.mean_squared_error(boost, given_output[6])

    return [output, elements[1], elements[2], elements[3]]

def run():
    with tf.Session() as sess:
        formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, batch_size)
        packet_generator = r.TensorflowPacketGenerator(batch_size)
        output_creator = tutorial_bot_output.TutorialBotOutput(batch_size)
        actions = action_handler.ActionHandler(split_mode=True)

        model = tutorial_model.TutorialModel(sess, n_input, n_output, action_handler=actions, is_training=True)
        model.num_layers = 10
        model.summary_writer = tf.summary.FileWriter(
            model.get_event_path('random_packet'))
        model.batch_size = batch_size
        model.mini_batch_size = batch_size

        # start model construction
        input_state, game_tick_packet = get_random_data(packet_generator, formatter)

        real_output = output_creator.get_output_vector(game_tick_packet)

        real_indexes = actions.create_indexes_graph(tf.stack(real_output, axis=1))

        reshaped = tf.cast(real_indexes, tf.int32)
        model.taken_actions = reshaped
        model.create_model(input_state)
        model.create_reinforcement_training_model(input_state)

        model.create_savers()

        start = time.time()

        checks = controller_statistics.OutputChecks(batch_size, model, sess, actions)

        model.initialize_model()

        # untrained bot
        checks.get_amounts()

        # RUNNING
        for i in tqdm(range(total_batches)):
            result, summaries = sess.run([model.train_op,
                      model.summarize if model.summarize is not None else model.no_op])

            if model.summary_writer is not None:
                model.summary_writer.add_summary(summaries, i)
            if ((i + 1) * batch_size) % 100000 == 0:
                model.save_model(model.get_model_path(model.get_default_file_name()))
        model.save_model(model.get_model_path(model.get_default_file_name()))

        total_time = time.time() - start
        print('total time: ', total_time)
        print('time per batch: ', total_time / (float(total_batches)))

        checks = controller_statistics.OutputChecks(batch_size, model, sess, actions)
        checks.get_amounts()

if __name__ == '__main__':
    run()
