import tensorflow as tf
from layers import fully_connected, unfold
from tensorflow.contrib.rnn import LSTMCell

class LSTM_classifier:
    def __init__(self, seq_len, input_length):
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, seq_len, input_length), name="input")
        y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="labels")

        multi_layer_cell = tf.contrib.rnn.MultiRNNCell([LSTMCell(n_neurons) for n_neurons in [64, 128]],
                                                       state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(multi_layer_cell, inputs, dtype=tf.float32, scope="lstm")

        x = unfold(x, "unfolded")

        x = fully_connected(x, 32, activation=tf.nn.relu, name="fc0")
        x = fully_connected(x, 16, activation=tf.nn.relu, name="fc1")
        x = fully_connected(x, 1, activation=None, name="fc2")

        self.x = inputs
        self.y = y
        self.pred = x
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.pred),
                                   name="loss")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.training_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-07).minimize(self.loss)
        self.correct = tf.cast(tf.equal(tf.cast(tf.greater(tf.nn.sigmoid(self.pred), 0.5), tf.float32), self.y),
                               tf.float32)
        self.accuracy = tf.reduce_mean(self.correct, name="accuracy")
