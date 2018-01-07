import tensorflow as tf
import time

from TutorialBot.atba2_demo_output import TutorialBotOutput_2
from TutorialBot.tutorial_bot_output import TutorialBotOutput
from conversions.input import tensorflow_input_formatter
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from trainer.utils import random_packet_creator as r
from models.actor_critic import tutorial_model
from modelHelpers.actions import action_handler
from trainer.utils import controller_statistics
from tqdm import tqdm


def get_random_data(packet_generator, input_formatter):
    game_tick_packet = packet_generator.get_random_array()
    output_array = input_formatter.create_input_array(game_tick_packet, game_tick_packet.time_diff)
    # reverse the shape of the array
    return output_array, game_tick_packet


learning_rate = 0.01
total_batches = 4000
batch_size = 5000
save_step = 2000000

# Network Parameters
n_neurons_hidden = 128  # every layer of neurons
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
        feature_creator = TensorflowFeatureCreator()
        formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, batch_size, feature_creator)
        packet_generator = r.TensorflowPacketGenerator(batch_size)
        output_creator = TutorialBotOutput(batch_size)
        actions = action_handler.ActionHandler(split_mode=True)

        model = tutorial_model.TutorialModel(sess, formatter.get_state_dim_with_features(),
                                             n_output, action_handler=actions, is_training=True)
        model.num_layers = 10
        model.summary_writer = tf.summary.FileWriter(
            model.get_event_path('random_packet'))
        model.batch_size = batch_size
        model.mini_batch_size = batch_size

        # start model construction
        input_state, game_tick_packet = get_random_data(packet_generator, formatter)

        real_output = output_creator.get_output_vector(game_tick_packet)

        real_indexes = actions.create_action_indexes_graph(tf.stack(real_output, axis=1))

        reshaped = tf.cast(real_indexes, tf.int32)
        model.taken_actions = reshaped
        model.create_model(input_state)
        model.create_reinforcement_training_model(input_state)

        model.create_savers()

        checks = controller_statistics.OutputChecks(batch_size, model.argmax, game_tick_packet,
                                                    input_state, sess, actions, output_creator)
        model.initialize_model()

        checks.create_model()

        # untrained bot
        start = time.time()
        checks.get_amounts()
        #print('time to get stats', time.time() - start)
        #for i in tqdm(range(total_batches)):
        #    sess.run([model.train_op])

        print_every_x_batches = (total_batches * batch_size) / save_step
        print('prints at this percentage', 100.0 / print_every_x_batches)
        model_counter = 0
        # RUNNING
        for i in tqdm(range(total_batches)):
            result, summaries = sess.run([model.train_op,
                      model.summarize if model.summarize is not None else model.no_op])

            if model.summary_writer is not None:
                model.summary_writer.add_summary(summaries, i)
            if ((i + 1) * batch_size) % save_step == 0:
                print()
                print('stats at', (i + 1) * batch_size, 'frames')
                checks.get_amounts()
                print('saving model')
                model.save_model(model.get_model_path(model.get_default_file_name() + str(model_counter)))
                model_counter += 1

        model.save_model(model.get_model_path(model.get_default_file_name()))

        total_time = time.time() - start
        print('total time: ', total_time)
        print('time per batch: ', total_time / (float(total_batches)))

        print('final stats')
        checks.get_amounts()
        checks.get_final_stats()

if __name__ == '__main__':
    run()
