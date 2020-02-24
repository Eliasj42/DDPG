import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import keras.backend as K


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    state_input = Input(shape=[state_size])
    layer0 = Dense(400, activation='relu')(state_input)
    layer1 = Dense(400, activation='relu')(layer0)
    value = Dense(action_size, activation = 'linear')(layer1)
    model = tf.keras.Model(inputs=state_input, outputs=value)
    return model, model.trainable_weights, state_input



class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.tau = tau
        self.model, self.weights, self.state = create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """

        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
