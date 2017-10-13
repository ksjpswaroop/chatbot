# coding: utf-8
import pandas as pd
import numpy as np
import re
import time
import collections
import os
from bz2 import BZ2File
from io import open

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))
from sklearn.model_selection import train_test_split

X, y = get_dataset_classification()

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 42)

print("training set:", X_train.shape)
print("test set:", X_test.shape)

# classes in this problem:
# conversation
# greetings
# calculation
# others
#
# modeling -- classification for modeling 
# SET Hyperparams at first !!!
learning_rate = 0.005
learning_rate_decay = 0.95
min_learning_rate = 0.0005
epochs = 100
batch_size = 32
keep_probability = 0.8
# 1 - GradientDescentOptimizer
# 2 - AdamOptimizer
# 3 - RMSPropOptimizer
model_optimizer = 1

# Hyperparams for cells
# 1 - Basic RNN
# 2 - GRU
# 3 - LSTM
encoder_cell_type = 2
rnn_dim = 512
encoder_forget_bias = 1.0

# 1 - tf.random_uniform_initializer
# 2 - tf.truncated_normal_initializer
# 3 - tf.orthogonal_initializer
initializer_type = 3

activation = 'tanh' # tanh, relu, etc....
num_layers = 1

# # Hyperparams for attentions
# # 1 - tf.contrib.seq2seq.BahdanauAttention()
# # 2 - tf.contrib.seq2seq.LuongAttetion()
# attention_type = 1

# others 
# gradient clipping
model_gradient_clipping = 1
clip_value_min = -5
clip_value_max = 5
clip_norm = 10

def model_inputs():

	input_data = tf.placeholder(tf.int32, [None, None], name='input')
	targets = tf.placeholder(tf.int32, [None, None], name='targets')
	lr = tf.placeholder(tf.float32, name='learning_rate')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
	max_summary_length = tf.reduce_max(summary_length, name='max_decoder_len')
	text_length = tf.placeholder(tf.int32, (None,), name='text_length')

	return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, word_to_int, batch_size):
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], word_to_int['<GO>']), ending], axis=1)
    return decoder_input


def encoding_layer(rnn_dim, sequence_length, num_layers, rnn_inputs, keep_prob):
    
    # choose the cell type to use
    if encoder_cell_type == 1:
        tf_cell = tf.contrib.rnn.RNNCell
    elif encoder_cell_type == 2:
        tf_cell = tf.contrib.rnn.GRUCell
    else:
        tf_cell = tf.contrib.rnn.LSTMCell
    # choose the initializer to use in cells
    if initializer_type == 1:
        initializer = tf.random_uniform_initializer(1.0, 1.0, seed=2)
    elif initializer_type == 2:
        initializer = tf.truncated_normal_initializer(1.0, 1.0, seed=2)
    else:
        initializer = tf.orthogonal_initializer(gain=1.0, seed=2)

    # stacking the cells with 2 directions
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf_cell(rnn_dim,
                initializer=initializer,
                forget_bias = encoder_forget_bias, 
                activation = activation)
            # add dropout wrapper
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                input_keep_prob = keep_prob)

            cell_bw = tf_cell(rnn_dim,
                initializer=initializer,
                forget_bias = encoder_forget_bias, 
                activation = activation)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                input_keep_prob = keep_prob)
            
            # add bidirectional wrapper
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    encoder_output = tf.concat(encoder_output,2)
    
    return encoder_output, encoder_state




