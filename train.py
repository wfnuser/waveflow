import os
from os.path import join
import glob

import numpy as np
import tensorflow as tf
from config import *
from rnn import RNN
from prepare_features import get_files_and_que, get_batch
import time

os.environ['CUDA_DEVICE_ORDER'] = '1'
start = time.clock()

# tf Graph input
X = tf.placeholder("float", [None, None, SP_DIM])
Y = tf.placeholder("float", [None, num_classes])
print("after initialization :", time.clock() - start)


print("after define placeholder :", time.clock() - start)

logits = RNN(X)
prediction = tf.nn.softmax(logits)
print("after define network :", time.clock() - start)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
print("after reduce_mean :", time.clock() - start)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
print("after gradientde :", time.clock() - start)
train_op = optimizer.minimize(loss_op)
print("after define loss :", time.clock() - start)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("after evaluate model :", time.clock() - start)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
print("after initialization :", time.clock() - start)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    print("after load session :", time.clock() - start)

    for epoch in range(1, 1000):
        print("It's epoch ", epoch, ", Time is :", time.clock() - start)
        file_pattern = "./dataset/vcc2016/bin/Training Set/*/1000*.bin"
        files, filename_queue = get_files_and_que(file_pattern)
        # print("After get files ", ", Time is :", time.clock() - start)
        for step in range(0, len(files) / batch_size + 1):
            # print("It's step ", step, ", Time is :", time.clock() - start)
            x, y, length = get_batch(files, step, batch_size)
            # print("After get batch ", ", Time is :", time.clock() - start)
            # Run optimization op (backprop)
            for truncatedStep in range(0, length / truncated_backprop_length):
                sess.run(train_op, feed_dict={X: x[:,truncatedStep*truncated_backprop_length:truncatedStep*truncated_backprop_length+truncated_backprop_length], Y:y})
            # print("After run sess ", ", Time is :", time.clock() - start)
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x, Y: y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))




    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, FEAT_DIM))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
