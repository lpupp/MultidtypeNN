"""NN where dtype precision is increased during training."""

import tensorflow as tf


def dtype2str(dtype):
    return str(dtype).split(' ')[1][1:-2]


def cast_params(params, dtype):
    cast_params = {}
    for layer in params:
        print('casting layer {}'.format(layer))
        cast_params[layer] = tf.cast(params[layer], dtype)
    return cast_params


def cast_assign_params(from_params, to_params, dtype):
    for layer in from_params:
        print('casting layer {}'.format(layer))
        to_params[layer] = tf.assign(tf.cast(from_params[layer], dtype))


def read_data(path):
    pass
