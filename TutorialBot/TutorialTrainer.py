import tensorflow as tf
import time
from TutorialBot import tutorial_bot_output
from TutorialBot import Randomizer as r
from conversions import input_formatter_no_rewards

start_time = time.time()

rand = r.PacketGenerator()

form = input_formatter_no_rewards.InputFormatter(0, 0)

tutorial_bot_output.get_output_vector(rand.get_random_packet(), [1, 1, 1, 1, 1, 1, 1.0, 1.0])

for n in range(1):
    learning_rate = 0.3
    total_batches = 1
    batch_size = 1
    display_step = 1

    # Network Parameters
    n_neurons_hidden = 128  # every layer of neurons
    n_input = 198  # data input
    n_output = 8  # total outputs

    input_state = tf.placeholder(tf.float32, [None, n_input])
    calculated_loss = tf.placeholder(tf.float32, [None, n_output])

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


    # Construct model
    logits = multilayer_perceptron(input_state)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(calculated_loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        avg_cost = 0.
        for i in range(total_batches):
            batch_x = []
            batch_y = []
            for m in range(batch_size):
                packet = rand.get_random_packet()
                x = form.create_input_array(packet)[0]
                batch_x.append(x)
                batch_y.append(tutorial_bot_output.get_output_vector(packet, multilayer_perceptron(x)))

            _, c = sess.run([train_op, loss_op], feed_dict={input_state: batch_x,
                                                            calculated_loss: batch_y})
            # Compute average loss
            avg_cost += c / batch_size
            # Display logs per epoch step
            print("Cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

elapsed_time = time.time() - start_time
print(elapsed_time)
