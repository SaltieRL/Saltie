import tensorflow as tf
import time

from trainer.base_classes.default_model_trainer import DefaultModelTrainer
from trainer.utils import random_packet_creator
from trainer.utils import controller_statistics
from tqdm import tqdm

from trainer.utils.trainer_runner import run_trainer


class RandomPacketTrainer(DefaultModelTrainer):
    total_batches = None
    save_step = None
    teacher_package = None
    teacher = None
    controller_stats = None
    start_time = None
    model_save_time = None
    frame_per_file = 20000

    def __init__(self):
        super().__init__()

    def get_random_data(self, packet_generator, input_formatter):
        game_tick_packet = packet_generator.get_random_array()
        output_array = input_formatter.create_input_array(game_tick_packet, game_tick_packet.time_diff)
        return output_array, game_tick_packet

    def get_config_name(self):
        return 'randomised_trainer.cfg'

    def get_event_filename(self):
        return 'random_packet'

    def load_config(self):
        super().load_config()
        # Obtaining necessary data for training from the config
        config = self.create_config()
        self.total_batches = config.getint('Randomised Trainer Configuration', 'total_batches')
        self.save_step = config.getint('Randomised Trainer Configuration', 'save_step')
        # Over here the model data is obtained
        self.teacher_package = config.get('Randomised Trainer Configuration', 'teacher_package')

    def setup_trainer(self):
        super().setup_trainer()
        self.teacher = self.teacher_package.split('.')[-1]

    def instantiate_model(self, model_class):
        return model_class(self.sess, self.action_handler.get_logit_size(),
                           action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_config(), teacher=self.teacher)

    def setup_model(self):
        super().setup_model()
        output_creator = self.get_class(self.teacher_package, 'TutorialBotOutput')(self.batch_size)
        packet_generator = random_packet_creator.TensorflowPacketGenerator(self.batch_size)
        input_state, game_tick_packet = self.get_random_data(packet_generator, self.input_formatter)

        real_output = output_creator.get_output_vector(game_tick_packet)
        real_indexes = self.action_handler.create_action_indexes_graph(tf.stack(real_output, axis=1))
        self.model.create_model(input_state)
        self.model.create_copy_training_model(model_input=input_state, taken_actions=real_indexes)
        self.model.create_savers()
        self.model.initialize_model()

        # Print out what the model uses
        self.model.printParameters()

        # Initialising statistics and printing them before training
        self.controller_stats = controller_statistics.OutputChecks(self.sess, self.action_handler,
                                                                   self.batch_size, self.model.smart_max,
                                                                   game_tick_packet=game_tick_packet,
                                                                   bot=output_creator)
        self.controller_stats.create_model()

    def _run_trainer(self):
        self.start_time = time.time()
        self.controller_stats.get_amounts()

        total_batches = self.total_batches
        batch_size = self.batch_size
        save_step = 100.0 / self.save_step
        sess = self.sess
        model = self.model

        # Percentage to print statistics (and also save the model)
        save_step = (total_batches * batch_size) / save_step
        print('training on the equivalent of', self.total_batches * self.batch_size / self.frame_per_file, 'games')
        print('Prints at this percentage:', 100.0 / save_step)
        model_counter = 0
        self.model_save_time = 0

        # Running the model
        for i in tqdm(range(total_batches)):
            model.run_train_step(True, None, i)

            if ((i + 1) * batch_size) % save_step == 0:
                print('\nStats at', (i + 1) * batch_size, 'frames (', i + 1, 'batches): ')
                self.controller_stats.get_amounts()
                print('Saving model', model_counter)
                start_saving = time.time()
                model.save_model(model.get_model_path(model.get_default_file_name() + str(model_counter)),
                                 global_step=i, quick_save=True)
                # print('saved model in', time.time() - start_saving, 'seconds')
                self.model_save_time += time.time() - start_saving
                model_counter += 1

    def finish_trainer(self):
        print('trained on the equivalent of', self.total_batches * self.batch_size / self.frame_per_file, 'games')
        start_saving = time.time()
        self.model.save_model()
        print('saved model in', time.time() - start_saving, 'seconds')
        self.model_save_time += time.time() - start_saving

        total_time = time.time() - self.start_time
        print('Total time:', total_time)
        print('Time per batch:', (total_time - self.model_save_time) / (float(self.total_batches)))
        print('Time spent saving', self.model_save_time)

        print('Final stats:')
        self.controller_stats.get_amounts()
        self.controller_stats.get_final_stats()
        super().finish_trainer()


if __name__ == '__main__':
    run_trainer(trainer=RandomPacketTrainer())
