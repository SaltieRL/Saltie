import tensorflow as tf
import time
from TutorialBot import TutorialBotOutput
from TutorialBot import RandomTFArray as r

def get_random_data(batch_size):
  array = r.get_random_array()
  # reverse the shape of the array
  inverted_array = tf.reshape(array, [tf.shape(array)[1], batch_size]
  game_tick_packet = output_formatter.create_output_array(array)
  return inverted_array, game_tick_packet


def get_loss(logits, game_tick_packet):
  return TutorialBotOutput.get_output_vector(logits, game_tick_packet)
  
learning_rate = 0.3
total_batches = 1
batch_size = 1
display_step = 1

# Network Parameters
n_neurons_hidden = 128  # every layer of neurons
n_input = 198  # data input
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
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    return out_layer


#start model construction
input_state, game_tick_packet = get_random_data(batch_size)

logits = multilayer_perceptron(input_state)

loss_op = get_loss(logits, game_tick_packet)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    avg_cost = 0.
    for i in range(total_batches):
      _, c = sess.run([train_op, loss_op])
      # Compute average loss
      avg_cost += c / batch_size
      # Display logs per epoch step
      print("Cost={:.9f}".format(avg_cost))
