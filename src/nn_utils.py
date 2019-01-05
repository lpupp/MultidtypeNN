"""NN where dtype precision is increased during training."""

import tensorflow as tf

def weight_variable(w_dim, name, trainable=True, dtype):
    """Create weights.

    Args:
        w_dim (list): [input dimension, output dimension]
    """
    t_stnd = (tf.sqrt(tf.cast(w_dim[0], dtype)) * 1000)
    return tf.Variable(tf.random_normal(w_dim, dtype=dtype) / t_stnd, name=name, trainable=trainable)


def bias_variable(b_dim, name, trainable=True, dtype):
    """Create biases."""
    t_stnd = tf.cast(b_dim[0], dtype) * 1000
    return tf.Variable(tf.random_normal(b_dim, dtype=dtype) / t_stnd, name=name, trainable=trainable)


def init_param(n_nodes, params, trainable=True, dtype=tf.int32):
    n_layers = len(n_nodes)

    x = {}
    if params == 'weights':
        _var = weight_variable
        _dim = lambda i: [n_nodes[i], n_nodes[i+1]]
    elif params == 'biases':
        _var = bias_variable
        _dim = lambda i: [n_nodes[i+1]]
    else:
        NotImplementedError

    for i in range(0, n_layers - 1):
        nm = params[0] + str(i)
        with tf.name_scope(params):
            [n_nodes[i+1]]
            x[i] = _var(_dim(i), name=nm, trainable=trainable, dtype=dtype)
            if tb_flag:
                tf.summary.histogram(nm, x[i])
    return x
