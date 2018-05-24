import tensorflow as tf
from config import *
from tensorflow.contrib import rnn
import time

def rnn_softmax(outputs, scope):
    with tf.variable_scope(scope, reuse=False):
        W_softmax = tf.get_variable("W_softmax",
            [num_hidden*2, DIM])
        b_softmax = tf.get_variable("b_softmax", [DIM])

    logits = tf.matmul(outputs, tf.tile(tf.expand_dims(W_softmax,0),[batch_size,1,1])) + b_softmax
    return logits

class style_trasfer_net(object):

    def build(self):
        last_time = time.clock()

        # Placeholders
        self.encoder_inputs_s = tf.placeholder(tf.float32, shape=[batch_size, None, DIM],
            name='encoder_inputs_s')
        self.encoder_inputs_c = tf.placeholder(tf.float32, shape=[batch_size, None, DIM],
            name='encoder_inputs_c')
        self.decoder_inputs = tf.placeholder(tf.float32, shape=[batch_size, None, DIM],
            name='decoder_inputs')
        self.targets = tf.placeholder(tf.float32, shape=[batch_size, None, DIM],
            name='targets')

        print("init placeholders cost: " + str(time.clock() - last_time))
        last_time = time.clock()

        # style encoder
        with tf.variable_scope('style_encoder') as scope:
            lstm_cell = rnn.LSTMCell(num_hidden, forget_bias=1.0)
            self.style_outputs, self.style_state = tf.nn.dynamic_rnn(lstm_cell, self.encoder_inputs_s, dtype=tf.float32)

        print("init style encoder cost: " + str(time.clock() - last_time))
        last_time = time.clock()

        # content encoder
        with tf.variable_scope('content_encoder') as scope:

            lstm_cell = rnn.LSTMCell(num_hidden, forget_bias=1.0)
            self.content_outputs, self.content_state = tf.nn.dynamic_rnn(lstm_cell, self.encoder_inputs_c, dtype=tf.float32)

        print("init content encoder cost: " + str(time.clock() - last_time))
        last_time = time.clock()

        print(self.style_state)
        # print(self.style_state[0].get_shape())

        with tf.variable_scope('decoder') as scope:
            # Initial state is last relevant state from encoder
            self.decoder_initial_state = tf.concat([self.style_state[0], self.content_state[0]],1)
            lstm_cell = rnn.LSTMCell(num_hidden*2, forget_bias=1.0)
            self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(lstm_cell, self.decoder_inputs, dtype=tf.float32)

        print("init decoder cost: " + str(time.clock() - last_time))
        last_time = time.clock()

        self.generate = rnn_softmax(self.decoder_outputs,'fc')
        self.loss = tf.reduce_mean(tf.square(self.generate - self.targets))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train_optimizer = optimizer.minimize(self.loss)

        print("init optimizer cost: " + str(time.clock() - last_time))
        last_time = time.clock()







        # print("step cost: " + str(time.clock() - last_time))

