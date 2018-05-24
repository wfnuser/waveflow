import os
from os.path import join
import glob

import numpy as np
import tensorflow as tf
from config import *
from prepare_features import get_files_and_que, get_batch
import time
import random
import model
from utils import shift_and_pad_zero


os.environ['CUDA_DEVICE_ORDER'] = '1'
start = time.clock()

network = model.style_trasfer_net()
network.build()
# Start training
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(tf.global_variables_initializer())
    variables = tf.trainable_variables()
    for v in variables:
        print(v)

    style_dir = "../../musicbin/training_data/dull_bell/"
    content_dir = "../../musicbin/training_data/piano/"
    target_dir = "../../musicbin/training_data/target/"

    for epoch in range(100):
        seq = range(30)
        random.shuffle(seq)
        for step in range(0, len(seq) / batch_size + 1):
            batch_encoder_inputs_s, batch_encoder_inputs_c, batch_targets, length = get_batch(seq, content_dir, style_dir, target_dir, step, batch_size)

            # to generate batch decoder intpus form batch targets
            batch_decoder_inputs = np.roll(batch_targets, 1, 1)
            for i in range(batch_decoder_inputs.shape[0]):
                np.put(batch_decoder_inputs, range(i*batch_decoder_inputs.shape[1]*batch_decoder_inputs.shape[2], i*batch_decoder_inputs.shape[1]*batch_decoder_inputs.shape[2] + batch_decoder_inputs.shape[2]), 0)

            input_feed = {network.encoder_inputs_s: batch_encoder_inputs_s,
                network.encoder_inputs_c: batch_encoder_inputs_c,
                network.decoder_inputs: batch_decoder_inputs,
                network.targets: batch_targets}

            train_op = network.train_optimizer

            _ = sess.run(train_op, input_feed)

            if step%display_step == 0:
                loss = sess.run(network.loss, input_feed)
                print("epoch: ", epoch, "step: ", step, "loss:", loss)

            if step % save_step == 0:
                saver.save(sess, './model/', meta_graph_suffix=str(epoch)+'-'+str(step)+'-')


    print("Optimization Finished!")

