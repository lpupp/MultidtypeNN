"""NN where dtype precision is increased during training."""

import tensorflow as tf


def dtype2str(dtype):
    return str(dtype).split(' ')[1][1:-2]


def cast_params(params, dtype):
    # TODO (@lpupp) do I need to store the cast? or do i need to copy?
    cast_params = {}
    for layer in params:
        print('casting layer {}'.format(layer))
        cast_params[layer] = tf.cast(params[layer], dtype)
    return cast_params


def cast_assign_params(from_params, to_params, dtype):
    # TODO (@lpupp) do I need to store the cast? or do i need to copy?
    for layer in from_params:
        print('casting layer {}'.format(layer))
        to_params[layer] = tf.assign(tf.cast(from_params[layer], dtype))
