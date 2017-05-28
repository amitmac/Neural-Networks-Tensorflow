import tensorflow as tf

def _lstm_cell_forward_prop(x, cell_prev, h_prev, W_forget, b_forget,
                            W_input, b_input, W_cell, b_cell, W_output, b_output):
    """
    LSTM forward propagation for one time step
        -   f_t = sigmoid(W_f.[h_{t-1}, x_t] + b_f)
        -   i_t = sigmoid(W_i.[h_{t-1}, x_t] + b_i)
        -   C_{t}^' = tanh(W_C.[h_{t-1}, x_t] + b_C)
        -   C_{t-1}^' = C_{t-1}*ft 
        -   C_t = C_{t-1}^' + i_{t} * C_t
        -   o_t = sigmoid(W_o * [h_{t-1},x_t] + b_o)
        -   h_t = o_t * tanh(C_t)

    args:
        x: input of size (batch_size, input_size)
        cell_prev: cell state at previous time step
        h_prev: hidden state at previous time step
        W_forget: weight matrix for forget gate
        b_forget: bias vector for forget gate

    return:
        cell_state: new cell state
        hidden_state: new hidden state
    """

    x_h_prev = tf.concat([x, h_prev], axis=1) # (batch_size, input_size + hidden_units)
    
    forget_state = tf.sigmoid(tf.add(tf.matmul(x_h_prev, W_forget), b_forget)) # (batch_size, hidden_units)
    input_state = tf.sigmoid(tf.add(tf.matmul(x_h_prev, W_input), b_input)) # (batch_size, hidden_units)

    cell_state = tf.nn.tanh(tf.add(tf.matmul(x_h_prev, W_cell), b_cell))
    cell_prev = tf.multiply(forget_state, cell_prev)
    cell_state = tf.add(cell_prev, tf.multiply(input_state, cell_state))

    output_state = tf.sigmoid(tf.add(tf.matmul(x_h_prev, W_output), b_output))
    hidden_state = tf.multiply(output_state, tf.nn.tanh(cell_state))

    return cell_state, hidden_state

def _gru_cell_forward_prop():
    """
    GRU forward propagation for one time step
    """
    pass