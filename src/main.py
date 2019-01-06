"""NN where dtype precision is increased during training."""

import os
import sys

import numpy as np
import tensorflow as tf

from model import MultidtypeNN
from utils import read_data




if __name__ == "__main__":
    sess = tf.Session()

    read_data()
    
    sys.argv[0]

    mdtnn = MultidtypeNN(n_nodes=[5, 64, 64, 64, 1],
                         activation_list=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
                         weight_init_method=None,
                         bias_init_method=None,
                         trainable=True,
                         with_bn=False,
                         dropout_rate=0,
                         tb_flag=False,
                         dtypes=[tf.float16, tf.float32])

    mdtnn.train_init(dim_input=5,
                     y=y,
                     learning_rate=0.3,
                     opt_nm='sgd',
                     cost_nm='mse',
                     batch_size=32,
                     n_epochs_per_precision=200)

    mdtnn.train(sess, x)

    preds = mdtnn.predict(x, batch_size=32)
    print(sess.run(preds))
