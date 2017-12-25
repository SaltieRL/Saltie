import tensorflow as tf
import TutorialBot
import time
import Randomizer as r
import input_formatter

b = TutorialBot.Agent("TestBot", 0, 0)

start_time = time.time()

rand = r.PacketGenerator()

form = input_formatter.InputFormatter(0, 0)

for n in range(1):
    # X = input_of_game_in_vector
    # Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    # Y = tf.nn.softmax(tf.matmul(Y1, W2) + B2)
    # Parameters
    learning_rate = 0.3
    total_batches = 15
    batch_size = 1000
    display_step = 1

    # Network Parameters
    n_neurons_hidden = 128  # every layer of neurons
    n_input = 198  # data input
    n_classes = 8  # total classes

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_neurons_hidden])),
        # tf.truncated_normal([n_input, n_neurons_hidden], stddev=0.1)),
        'h2': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
        'h3': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
        'h4': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
        'h5': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
        'out': tf.Variable(tf.random_normal([n_neurons_hidden, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_neurons_hidden])),
        'b2': tf.Variable(tf.random_normal([n_neurons_hidden])),
        'b3': tf.Variable(tf.random_normal([n_neurons_hidden])),
        'b4': tf.Variable(tf.random_normal([n_neurons_hidden])),
        'b5': tf.Variable(tf.random_normal([n_neurons_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 48 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 24 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
        return out_layer


    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
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
                batch_x.append(form.create_input_array(packet)[0])
                batch_y.append(b.get_output_vector(packet))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / batch_size
            # Display logs per epoch step
            print("Cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

elapsed_time = time.time() - start_time
print(elapsed_time)
