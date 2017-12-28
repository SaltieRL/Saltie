import tensorflow as tf
import time
from conversions import input_formatter
from TutorialBot import tensorflow_input_formatter
from TutorialBot import tutorial_bot_output
from TutorialBot import RandomTFArray as r
from models.actor_critic import base_actor_critic
from modelHelpers import action_handler


def get_random_data(batch_size, packet_generator, input_formatter):
  game_tick_packet = packet_generator.get_random_array(batch_size)
  output_array = input_formatter.create_input_array(game_tick_packet)[0]
  # reverse the shape of the array
  return output_array, game_tick_packet


def get_loss(logits, game_tick_packet, output_creator):
  return output_creator.get_output_vector(game_tick_packet, logits)

learning_rate = 0.1
total_batches = 100
batch_size = 1000
display_step = 1

# Network Parameters
n_neurons_hidden = 128  # every layer of neurons
n_input = input_formatter.get_state_dim_with_features()  # data input
n_output = 8  # total outputs

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_neurons_hidden])),
    'h2': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
    'h3': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
    'h4': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
    'h5': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
    'out': tf.Variable(tf.random_normal([n_neurons_hidden, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_neurons_hidden])),
    'b2': tf.Variable(tf.random_normal([n_neurons_hidden])),
    'b3': tf.Variable(tf.random_normal([n_neurons_hidden])),
    'b4': tf.Variable(tf.random_normal([n_neurons_hidden])),
    'b5': tf.Variable(tf.random_normal([n_neurons_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

 # Create model
def multilayer_perceptron(x):
    # 5 hidden layers with 128 neurons each
    layer_1 = tf.nn.relu6(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu6(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.relu6(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_4 = tf.nn.relu6(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    layer_5 = tf.nn.relu6(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.sigmoid(tf.matmul(layer_5, weights['out']) + biases['out'])

    return out_layer

if __name__ == '__main__':

    with tf.Session() as sess:

        formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, batch_size)
        packet_generator = r.TensorflowPacketGenerator(batch_size)
        output_creator = tutorial_bot_output.TutorialBotOutput(batch_size)
        #actions = action_handler.ActionHandler(split_mode=True)

        #model = base_actor_critic.BaseActorCritic(sess, n_input, actions)

        #start model construction
        input_state, game_tick_packet = get_random_data(batch_size, packet_generator, formatter)

        logits = multilayer_perceptron(input_state)

        loss_op = get_loss(logits, game_tick_packet, output_creator)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init = tf.global_variables_initializer()

        start = time.time()

        # RUNNING
        sess.run(init)
        # Training cycle
        cost = 0.0
        for i in range(total_batches):
            _, c = sess.run([train_op, tf.reduce_mean(loss_op)])
            # Compute average loss

            cost += c
            # Display logs per epoch step
            print("Cost =", cost / (i + 1))
        print("Final cost = ", cost / total_batches)
        total_time = time.time() - start
        print('total time', total_time)
