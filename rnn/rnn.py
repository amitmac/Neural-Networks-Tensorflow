import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

def dynamic_rnn(cell, inputs, initial_state=None, dtype=None, scope=None, time_major=False):
    """
    args:
        cell: cell type - LSTM/GRU
        inputs: input data
        time_major: If False, input_shape = (batch_size, max_time_steps, input_size)
                    If True,  input_shape = (max_time_steps, batch_size, input_size)
    """
    
    if not time_major:
        inputs = tf.transpose(inputs, perm=[1,0,2])

    max_time_steps, batch_size, input_size = inputs.get_shape().as_list()
    
    output_states = []

    with vs.variable_scope(scope or "rnn"):
        if initial_state==None:
            initial_state = cell.zero_state(batch_size, dtype)

        states = initial_state

        for time_step in range(max_time_steps):
            (cell_state, hidden_state) = cell(inputs[time_step], states)
            output_states.append(hidden_state)
            states = (cell_state, hidden_state)

    return (output_states, states)