import tensorflow as tf
import time

from bot_code.trainer.base_classes.default_model_trainer import DefaultModelTrainer
from bot_code.trainer.utils import random_packet_creator
from bot_code.trainer.utils import controller_statistics
from bot_code.utils.dynamic_import import get_class
from tqdm import tqdm



class RandomPacketTrainer(DefaultModelTrainer):
    total_batches = None
    save_step = None
    teacher_package = None
    teacher_class_name = None
    teacher = None
    controller_stats = None
    start_time = None
    model_save_time = None
    frame_per_file = 20000

    def __init__(self):
        super().__init__()

    def get_random_data(self, packet_generator, input_formatter):
        state_object = packet_generator.get_random_array()
        output_array = input_formatter.create_input_array(state_object, state_object.time_diff)
        return output_array, state_object

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
        self.teacher_class_name = config.get('Randomised Trainer Configuration', 'teacher_class_name')

    def setup_trainer(self):
        super().setup_trainer()
        self.teacher = self.teacher_package.split('.')[-1]

    def instantiate_model(self, model_class):
        return model_class(self.sess, self.action_handler.get_logit_size(),
                           action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_model_config(), teacher=self.teacher)

    def setup_model(self):
        super().setup_model()
        teacher_class = get_class(self.teacher_package, self.teacher_class_name)
        teacher = teacher_class(self.batch_size)
        packet_generator = random_packet_creator.TensorflowPacketGenerator(self.batch_size)
        input_state, state_object = self.get_random_data(packet_generator, self.input_formatter)

        real_output = teacher.get_output_vector_model(state_object)
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
                                                                   state_object=state_object,
                                                                   bot=teacher)
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
        save_step = int((total_batches * batch_size) / save_step)
        print('training on the equivalent of', self.total_batches * self.batch_size / self.frame_per_file, 'games')
        print('Prints at this percentage:', 100.0 / ((total_batches * batch_size) / save_step))
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
    RandomPacketTrainer().run()
