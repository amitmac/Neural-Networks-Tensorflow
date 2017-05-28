import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

import forward_prop

"""Recurrent Neural Networks implementation inspired by tensorflow"""

class LSTMCell():
    """
    LSTM Cell based on Chris Olah's Blog on Understanding LSTMs
    """
    def __init__(self, num_units, input_size):
        """
            args:
                num_units: number of hidden units/ state size
        """
        self._num_units = num_units
        self.W_forget = tf.Variable(tf.random_uniform([self._num_units + input_size, self._num_units],-1, 1), name="weight_forget")
        self.b_forget = tf.Variable(tf.zeros([self._num_units]), name="bias_forget")
        self.W_input = tf.Variable(tf.random_uniform([self._num_units + input_size, self._num_units],-1, 1), name="weight_input")
        self.b_input = tf.Variable(tf.zeros([self._num_units]), name="bias_input")
        self.W_cell = tf.Variable(tf.random_uniform([self._num_units + input_size, self._num_units],-1, 1), name="weight_cell")
        self.b_cell = tf.Variable(tf.zeros([self._num_units]), name="bias_cell")
        self.W_output = tf.Variable(tf.random_uniform([self._num_units + input_size, self._num_units],-1, 1), name="weight_output")
        self.b_output = tf.Variable(tf.zeros([self._num_units]), name="bias_output")

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """
            cell_state and hidden_state is of shape (batch_size, num_hidden_units)
        """
        cell_state = tf.zeros([batch_size, self._num_units], dtype=dtype)
        hidden_state = tf.zeros([batch_size, self._num_units], dtype=dtype)

        return (cell_state, hidden_state)

    def __call__(self, x, states_prev, scope=None):
        """
            args:
                x: input of size (batch_size, input_size)
                states_prev: (previous_cell_state, previous_hidden_state)
            return:
                cell_state, hidden_state 
        """
        if len(states_prev) != 2:
            raise ValueError("Expecting states_prev to be a tuple of length 2.")
        input_size = int(x.shape[1])
        
        cell_prev, h_prev = states_prev

        cell_state, hidden_state =  forward_prop._lstm_cell_forward_prop(x, cell_prev, h_prev, self.W_forget, self.b_forget,
                                                     self.W_input, self.b_input, self.W_cell, self.b_cell,
                                                     self.W_output, self.b_output)

        return cell_state, hidden_state

class GRUCell():
    """Basic LSTM Cell"""
    def __init__(self, num_units):
        """
            args:
                num_units: number of hidden units
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._names = {
            "W_cell": "weight_cell",
            "b_cell": "bias_cell",
            "W_output": "weight_output",
            "b_output": "bias_output",
            "scope": "gru_cell"
        }

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units 