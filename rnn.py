import tensorflow as tf
from tensorflow.contrib import rnn
from config import *
import time

def RNN(x):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Define weights
    start = time.clock()
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    print("initialize cost: ", time.clock()- start)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)


    # x = tf.unstack(x[:,500:600,:], 100, 1)
    # print("initialize unstack: ", time.clock()- start)


    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    print("initialize lstm: ", time.clock()- start)

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    print(outputs)
    print("initialize dynamic_rnn: ", time.clock()- start)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[:,-1], weights['out']) + biases['out']
