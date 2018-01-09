import tensorflow as tf
import time

import inspect
import importlib
import configparser

from TutorialBot.tutorial_bot_output import TutorialBotOutput
from conversions.input import tensorflow_input_formatter
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from trainer.utils import random_packet_creator as r
from modelHelpers.actions import action_factory, dynamic_action_handler
from trainer.utils import controller_statistics
from tqdm import tqdm


def get_random_data(packet_generator, input_formatter):
    game_tick_packet = packet_generator.get_random_array()
    output_array = input_formatter.create_input_array(game_tick_packet, game_tick_packet.time_diff)
    return output_array, game_tick_packet


def get_class(class_package, class_name):
    class_package = importlib.import_module(class_package)
    module_classes = inspect.getmembers(class_package, inspect.isclass)
    for class_group in module_classes:
        if class_group[0] == class_name:
            return class_group[1]
    return None


def run():
    # Obtaining necessary data for training from the config
    config = configparser.RawConfigParser()
    config.read('randomised_trainer.cfg')
    batch_size = config.getint('Randomised Trainer Configuration', 'batch_size')
    total_batches = config.getint('Randomised Trainer Configuration', 'total_batches')
    save_step = config.getint('Randomised Trainer Configuration', 'save_step')
    # Over here the model data is obtained
    model_package = config.get('Model Configuration', 'model_package')
    model_name = config.get('Model Configuration', 'model_name')
    model_class = get_class(model_package, model_name)
    num_layers = config.getint('Model Configuration', 'num_layers')

    with tf.Session() as sess:
        # Creating necessary instances
        feature_creator = TensorflowFeatureCreator()
        formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, batch_size, feature_creator)
        packet_generator = r.TensorflowPacketGenerator(batch_size)
        output_creator = TutorialBotOutput(batch_size)
        actions = action_factory.get_handler(control_scheme=dynamic_action_handler.super_split_scheme)

        # Initialising the model
        model = model_class(sess, formatter.get_state_dim_with_features(),
                            actions.get_logit_size(), action_handler=actions, is_training=True)
        model.num_layers = num_layers
        model.summary_writer = tf.summary.FileWriter(
            model.get_event_path('random_packet'))
        model.batch_size = batch_size
        model.mini_batch_size = batch_size

        # Starting model construction
        input_state, game_tick_packet = get_random_data(packet_generator, formatter)
        real_output = output_creator.get_output_vector(game_tick_packet)
        real_indexes = actions.create_action_indexes_graph(tf.stack(real_output, axis=1))
        reshaped = tf.cast(real_indexes, tf.int32)
        model.taken_actions = reshaped
        model.create_model(input_state)
        model.create_reinforcement_training_model(input_state)
        model.create_savers()
        model.initialize_model()

        # Print out what the model uses
        model.printParameters()

        # Initialising statistics and printing them before training
        checks = controller_statistics.OutputChecks(batch_size, model.argmax, game_tick_packet,
                                                    input_state, sess, actions, output_creator)
        checks.create_model()
        start = time.time()
        checks.get_amounts()

        # Percentage to print statistics (and also save the model)
        print_every_x_batches = (total_batches * batch_size) / save_step
        print('Prints at this percentage:', 100.0 / print_every_x_batches)
        model_counter = 0

        # Running the model
        for i in tqdm(range(total_batches)):
            result, summaries = sess.run([model.train_op,
                                          model.summarize if model.summarize is not None else model.no_op])

            if model.summary_writer is not None:
                model.summary_writer.add_summary(summaries, i)
            if ((i + 1) * batch_size) % save_step == 0:
                print('\nStats at', (i + 1) * batch_size, 'frames (', i + 1, 'batches): ')
                checks.get_amounts()
                print('Saving model', model_counter)
                model.save_model(model.get_model_path(model.get_default_file_name() + str(model_counter)))
                model_counter += 1

        final_model_path = model.get_model_path(model.get_default_file_name())
        model.save_model(final_model_path)

        total_time = time.time() - start
        print('Total time:', total_time)
        print('Time per batch:', total_time / (float(total_batches)))

        print('Final stats:')
        checks.get_amounts()
        checks.get_final_stats()


if __name__ == '__main__':
    run()
