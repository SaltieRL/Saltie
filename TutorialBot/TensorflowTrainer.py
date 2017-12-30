import tensorflow as tf
import time
from conversions import input_formatter
from TutorialBot import tensorflow_input_formatter
from TutorialBot import tutorial_bot_output
from TutorialBot import RandomTFArray as r
from models.actor_critic import base_actor_critic
from modelHelpers import action_handler


def get_random_data(packet_generator, input_formatter):
    game_tick_packet = packet_generator.get_random_array()
    output_array = input_formatter.create_input_array(game_tick_packet)[0]
    # reverse the shape of the array
    return output_array, game_tick_packet


def get_loss(logits, game_tick_packet, output_creator):
    return output_creator.get_output_vector(game_tick_packet, logits)


learning_rate = 0.1
total_batches = 50
batch_size = 1000
display_step = 1

# Network Parameters
n_neurons_hidden = 128  # every layer of neurons
n_input = input_formatter.get_state_dim_with_features()  # data input
n_output = 39  # total outputs

def create_loss(expected_outputs, created_outputs, logprobs):
    reshaped = tf.transpose(expected_outputs)
    reshaped = tf.cast(reshaped, tf.int32)
    output_yaw = reshaped[0]
    output_pitch = reshaped[1]
    output_roll = reshaped[2]
    output_button = reshaped[3]

    loss = tf.losses.absolute_difference(output_yaw, created_outputs[0], weights=0.5)
    loss += tf.losses.absolute_difference(output_pitch, created_outputs[1], weights=0.5)
    loss += tf.losses.absolute_difference(output_roll, created_outputs[2], weights=0.5)
    loss += tf.losses.mean_squared_error(output_button, created_outputs[3])


    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs[0],
                                                                        labels=output_yaw)
    cross_entropy_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs[1],
                                                                        labels=output_pitch)
    cross_entropy_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs[2],
                                                                         labels=output_roll)
    cross_entropy_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs[3],
                                                                         labels=output_button)

    return loss + cross_entropy_loss




if __name__ == '__main__':

    with tf.Session() as sess:
        formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, batch_size)
        packet_generator = r.TensorflowPacketGenerator(batch_size)
        output_creator = tutorial_bot_output.TutorialBotOutput(batch_size)
        actions = action_handler.ActionHandler(split_mode=True)

        model = base_actor_critic.BaseActorCritic(sess, n_input, n_output, action_handler=actions)

        # start model construction
        input_state, game_tick_packet = get_random_data(packet_generator, formatter)

        #logits = multilayer_perceptron(input_state)

        model.create_model(input_state)

        # the indexes
        print(model.argmax)
        created_actions = actions.create_tensorflow_controller_output_from_actions(model.argmax, batch_size)

        loss_op, real_output = get_loss(created_actions, game_tick_packet, output_creator)

        real_indexes = actions.create_indexes_graph(tf.stack(real_output, axis=1))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        loss_op += create_loss(real_indexes, model.argmax, model.softmax)



        train_op = optimizer.minimize(loss_op)
        init = tf.global_variables_initializer()

        start = time.time()

        model.batch_size = batch_size

        model.initialize_model()

        # RUNNING
        # Training cycle
        c = 0.
        for i in range(total_batches):
            _, c = sess.run([train_op, tf.reduce_mean(loss_op)])

            # Display logs per epoch step
            print("Cost: ", c)
        saver = tf.train.Saver()
        saver.save(sess, "./trained_variables/TensorflowTrainer.ckpt")
        print(sess.run(weights['out']))
        total_time = time.time() - start
        print('total time: ', total_time)
