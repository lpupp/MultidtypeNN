"""NN where dtype precision is increased during training."""

import tensorflow as tf
import numpy as np

# TODO (@lpupp) These may not be needed
txt2np_dtypes = {'int8': np.int8,
                 'int16': np.int16,
                 'int32': np.int32,
                 'int64': np.int64,
                 'float16': np.half,
                 'float32': np.single,
                 'float64': np.double}

txt2tf_dtypes = {'int8': tf.int8,
                 'int16': tf.int16,
                 'int32': tf.int32,
                 'int64': tf.int64,
                 'float16': tf.float16,
                 'float32': tf.float32,
                 'float64': tf.float64}

tf2np_dtypes = {tf.int8: np.int8,
                tf.int16: np.int16,
                tf.int32: np.int32,
                tf.int64: np.int64,
                tf.float16: np.half,
                tf.float32: np.single,
                tf.float64: np.double}


class NeuralNet(object):
    def __init__(self, n_nodes,
                 activation_list,
                 weights,
                 biases,
                 dtype,
                 with_bn=False,
                 dropout_rate=0):
        self.l_act = activation_list
        self.weights = weights
        self.biases = biases

        self.n_layers = len(n_nodes)
        print('{} hidden layers in NN'.format(self.num_layers-2))

        self.dtype = dtype

        self.with_bn = with_bn
        self.drop_rate = dropout_rate

    def __call__(self, x, training=True):
        for layer in range(0, self.n_layers - 1):
            x = tf.add(tf.matmul(x, self.weights[layer]), self.biases[layer])
            act = self.l_act[layer]
            if act is not None:
                x = act(x)
                # TODO (@lpupp) do we bn and do the output?
                if self.with_bn:
                    x = tf.layers.batch_normalization(x, training=training)
                if self.drop_rate:
                    x = tf.layers.dropout(x, rate=self.drop_rate, training=training)
        return x

    def predict(self, x, batch_size, training=False):
        out = []
        # TODO (@lpupp) check
        for i in range(0, x.shape[0] / batch_size):
            minibatch = x[i*batch_size:(i+1)*batch_size]
            out.append(self.__call__(minibatch, training=training))
        return np.concatenate(out, axis=1)  # TODO (@lpupp)


# TODO (@lpupp) Stick this in some model file
class MultidtypeNN(object):
    def __init__(self, n_nodes,
                 activation_list,
                 weight_init_method=None,
                 bias_init_method=None,
                 n_iterations_per_precision=1000,
                 trainable=True,
                 with_bn=False,
                 dropout_rate=0,
                 tb_flag=False,
                 dtypes=[tf.int8, tf.float16, tf.float32, tf.float64]):

        assert len(activation_list) == len(num_nodes) - 1, 'length of activation_list doesnt match the number of layers'

        self.n_nodes = n_nodes
        self.l_act = activation_list
        self.with_bn = with_bn
        self.drop_rate = dropout_rate
        self.w_init_method = weight_init_method
        self.b_init_method = bias_init_method

        self.n_layers = len(num_nodes)
        print('{} hidden layers in NN'.format(self.n_layers - 2))

        # Weights and biases keys are ints
        self.weights, self.biases = {}, {}
        self.NN = {}

        self.n_iter = n_iterations_per_precision

        self.dtypes = dtypes
        self.len_dtypes = len(dtypes)
        self.current_dtype = self.dtypes[0]

        self.tb_flag = tb_flag

        dtype_nm = dtype2str(self.current_dtype)
        self._init_params()
        self.NN[dtype_nm] = NeuralNet(self.n_nodes,
                                      self.l_act,
                                      self.weights[dtype_nm],
                                      self.biases[dtype_nm],
                                      self.current_dtype,
                                      with_bn=self.with_bn,
                                      dropout_rate=self.drop_rate)

    def _init_params(self):
        dtype_nm = dtype2str(self.current_dtype)
        with tf.name_scope('network_parameters'):
            if self.w_init_method is None and self.b_init_method is None:
                self.weights[dtype_nm] = init_param(self.n_nodes,
                                                    params='weights',
                                                    trainable=self.trainable,
                                                    dtype=self.current_dtype,
                                                    tb_flag=self.tb_flag)
                self.biases[dtype_nm] = init_param(self.n_nodes,
                                                   params='biases',
                                                   trainable=self.trainable,
                                                   dtype=self.current_dtype,
                                                   tb_flag=self.tb_flag)
            else:
                raise NotImplementedError

    def train(self, x, y):
        for _ in range(self.len_dtypes):
            for epoch in range(self.n_iter):
                self._train_epoch_dtype(x, y)
            self._update_current_dtype()
            self._cast_weights_2_next_dtype()
            self._cast_NN_2_next_dtype()

    def _train_epoch_dtype(self, x, y):
        # TODO(@lpupp) Train newest NN for a single epoch
        pass

    def _update_current_dtype(self):
        ix = self.dtypes.index(self.current_dtype)
        self.current_dtype = self.dtypes[ix + 1]

    def _cast_weights_2_next_dtype(self):
        dtype_nm = dtype2str(self.current_dtype)

        # Cast current weights to new dtype
        self.weights[dtype_nm] = cast_params(self.weights, self.current_dtype)
        self.biases[dtype_nm] = cast_params(self.biases, self.current_dtype)

    def _cast_NN_2_next_dtype(self):
        dtype_nm = dtype2str(self.current_dtype)

        self.NN[dtype_nm] = NeuralNet(self.n_nodes,
                                      self.l_act,
                                      self.weights[dtype_nm],
                                      self.biases[dtype_nm],
                                      self.current_dtype,
                                      with_bn=self.with_bn,
                                      dropout_rate=self.drop_rate)

    def predict(self, x, batch_size, training=False):
        out = []
        dtype_nm = dtype2str(self.current_dtype)
        nn = self.NN[dtype_nm]
        # TODO (@lpupp) check
        for i in range(0, x.shape[0] / batch_size):
            minibatch = x[i*batch_size:(i+1)*batch_size]
            out.append(nn(minibatch, training=training))
        return np.concatenate(out, axis=1)  # TODO (@lpupp)

    def __call__(self, x, training=True):
        dtype_nm = dtype2str(self.current_dtype)

        # TODO(@lpupp) is this needed?
        x = x.astype(tf2np_dtypes[self.current_dtype])
        nn = self.NN[dtype_nm]

        return nn(x)
