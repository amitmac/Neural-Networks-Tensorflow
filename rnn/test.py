import rnn_cells
from rnn import dynamic_rnn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""Taken from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767"""

num_epochs = 100
total_series_length = 50000
max_time_steps = 10
state_size = 10
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//max_time_steps//batch_size

def generate_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1, 1))
    y = y.reshape((batch_size, -1, 1))

    return (x, y)

data = generate_data()

batchX = tf.placeholder([batch_size, max_time_steps, 1], dtype=tf.float32)
batchY = tf.placeholder([batch_size, max_time_steps, 1], dtype=tf.int32)

inputs_series = tf.unpack(batchX, axis=1)
labels_series = tf.unpack(batchY, axis=1)

cell = rnn_cells.LSTMCell(state_size)

for epoch in xrange(num_epochs):
    for batch in range(num_batches):
        output_states, final_state = dynamic_rnn(cell, data[0], dtype=tf.float32)
