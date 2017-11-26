from models import nnatba
from modelHelpers import option_handler
import tensorflow as tf
import numpy as np

class NNAtbaTrainer:

    learning_rate = 0.3

    file_number = 0

    epoch = 0
    display_step = 1

    options = option_handler.createOptions()

    batch_size = 100
    input_batch = []
    label_batch = []

    def __init__(self):
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        # writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))

        self.state_dim = 195
        self.num_actions = len(self.options)
        self.agent = nnatba.NNAtba(self.sess, self.state_dim, self.num_actions)
        self.loss, self.input, self.label = self.agent.create_training_model_copy(batch_size=self.batch_size)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def start_new_file(self):
        self.file_number += 1
        self.input_batch = []
        self.label_batch = []

    def process_pair(self, input_array, output_array, pair_number):
        if len(self.input_batch) == self.batch_size:
            self.batch_process()
            self.input_batch = []
            self.label_batch = []
            # do stuff
        else:
            if len(input_array) == 193:
                input_array = np.append(input_array, [0])
                input_array = np.append(input_array, [0])
            self.input_batch.append(input_array)

            index = option_handler.find_matching_option(self.options, output_array)[1]
            array = np.zeros(self.num_actions)
            array[index] = 1
            self.label_batch.append(array)

    def batch_process(self):
     #   for i in range(len(self.input_batch)):
     #       one_input = self.input_batch[i]
     #       one_label = self.label_batch[i]

     #       one_input = one_input.reshape(1, len(one_input))
     #       one_label = np.array([one_label])
     #       one_label = one_label.reshape(1,)
        self.input_batch = np.array(self.input_batch)
        self.input_batch = self.input_batch.reshape(len(self.input_batch), self.state_dim)

        self.label_batch = np.array(self.label_batch)
        self.label_batch = self.label_batch.reshape(len(self.label_batch), self.num_actions)

        _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.input: self.input_batch, self.label: self.label_batch})
        # Display logs per step
        if self.epoch % self.display_step == 0:
            print("File:", '%04d' % self.file_number, "Epoch:", '%04d' % (self.epoch+1), "cost= " + str(c))
        self.epoch += 1

    def end_file(self):
        self.batch_process()
        if self.file_number % 3 == 0:
            saver = tf.train.Saver()
            saver.save(self.sess, "../models/data/trained_variables_drop" + str(self.file_number) + ".ckpt")
        pass

    def end_everything(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "../models/data/trained_variables_drop.ckpt")
