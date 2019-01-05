"""NN where dtype precision is increased during training."""

import tensorflow as tf


def dtype2str(dtype):
    return str(dtype).split(' ')[1][1:-2]


def cast_params(params, dtype):
    # TODO (@lpupp) do I need to store the cast?
    for layer in params:
        tf.cast(params[layer], dtype)
    return params
