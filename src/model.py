"""NN where dtype precision is increased during training."""

import tensorflow as tf
import numpy as np

from utils import dtype2str, cast_params
from nn_utils import init_param

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
        print('dtype {} NN has {} hidden layers'.format(dtype, self.n_layers-2))

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
        for i in range(0, int(x.shape[0]/batch_size)):
            minibatch = x[i*batch_size:(i+1)*batch_size]
            out.append(self.__call__(minibatch, training=training))
        return tf.concat(out, axis=0)  # TODO (@lpupp)


# TODO (@lpupp) Stick this in some model file
class MultidtypeNN(object):
    def __init__(self, n_nodes,
                 activation_list,
                 weight_init_method=None,
                 bias_init_method=None,
                 trainable=True,
                 with_bn=False,
                 dropout_rate=0,
                 tb_flag=False,
                 dtypes=[tf.float16, tf.float32, tf.float64]):

        assert len(activation_list) == len(n_nodes) - 1, 'length of activation_list doesnt match the number of layers'

        self.n_nodes = n_nodes
        self.l_act = activation_list
        self.with_bn = with_bn
        self.drop_rate = dropout_rate
        self.trainable = trainable
        self.w_init_method = weight_init_method
        self.b_init_method = bias_init_method

        self.n_layers = len(n_nodes)

        # Weights and biases keys are ints
        self.weights, self.biases = {}, {}
        self.NN = {}

        self.dtypes = dtypes
        self.len_dtypes = len(dtypes)
        self.current_dtype = self.dtypes[0]

        self.tb_flag = tb_flag
        self.train_inited = False
        #for dtype in self.dtypes:
        self._init_params()#dtype)
        self._init_NN()#dtype)

    def _init_NN(self, dtype=None):
        #dtype_nm = dtype2str(dtype)
        dtype_nm = dtype2str(self.current_dtype)
        self.NN[dtype_nm] = NeuralNet(self.n_nodes,
                                      self.l_act,
                                      self.weights[dtype_nm],
                                      self.biases[dtype_nm],
                                      self.current_dtype,
                                      with_bn=self.with_bn,
                                      dropout_rate=self.drop_rate)

    def _init_params(self, dtype=None):
        #dtype_nm = dtype2str(dtype)
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

    def _init_tf_vars(self):
        dtype_nm = dtype2str(self.current_dtype)
        self.y_pred[dtype_nm] = self.NN[dtype_nm](self.x[dtype_nm])

        self.cost[dtype_nm] = self.cost_fn(self.y_true[dtype_nm], self.y_pred[dtype_nm])

        self.grads[dtype_nm] = self.opt_fn.compute_gradients(self.cost[dtype_nm])
        self.train_step[dtype_nm] = self.opt_fn.apply_gradients(self.grads[dtype_nm])

    def train_init(self,
                   dim_input,
                   y,
                   learning_rate,
                   opt_nm,
                   cost_nm,
                   batch_size,
                   n_epochs_per_precision,
                   dtype=None):
        self.lr = learning_rate
        self.batch_size = batch_size
        if isinstance(n_epochs_per_precision, list):
            self.n_epochs = n_epochs_per_precision
        else:
            self.n_epochs = [n_epochs_per_precision]

        assert len(self.n_epochs) != 1 or len(self.n_epochs) != self.len_dtypes, 'dim of n_epochs and dtypes does not match'
        if len(self.n_epochs) == 1 and self.len_dtypes > 1:
            self.n_epochs = self.n_epochs * self.len_dtypes

        self.x, self.y_pred, self.y_true = {}, {}, {}
        self.cost, self.grads, self.train_step = {}, {}, {}

        if opt_nm == 'adam':
            self.opt_fn = tf.train.AdamOptimizer(learning_rate=learning_rate, name=opt_nm)
        elif opt_nm == 'sgd':
            self.opt_fn = tf.train.GradientDescentOptimizer(learning_rate, name=opt_nm)
        elif opt_nm == 'mom':
            self.opt_fn = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, name=opt_nm)
        elif opt_nm == 'nest_mom':
            self.opt_fn = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True, name=opt_nm)
        else:
            NotImplementedError

        if cost_nm == 'mse':
            self.cost_fn = lambda y_t, y_p: tf.reduce_mean((y_t - y_p) ** 2)
        else:
            NotImplementedError

        for dtype in self.dtypes:
            dtype_nm = dtype2str(dtype)

            self.x[dtype_nm] = tf.placeholder(dtype, shape=(None, dim_input), name='x'+dtype_nm)
            self.y_true[dtype_nm] = tf.constant(y, dtype=dtype, name='y_true'+dtype_nm)

        self._init_tf_vars()

        self.train_inited = True

    def train(self, sess, x):
        assert self.train_inited, 'Run MultidtypeNN.train_init(...) before training.'

        init = tf.global_variables_initializer()
        sess.run(init)

        for dt in range(self.len_dtypes):
            print('training {}'.format(self.current_dtype))
            for epoch in range(self.n_epochs[dt]):
                if epoch % 100 == 0:
                    print('training epoch {}/{}'.format(epoch, self.n_epochs[dt]-1))
                self._train_epoch_dtype(sess, x)
            if dt != self.len_dtypes-1:
                self._cast_params_2_next_dtype()
                self._update_current_dtype()
                self._cast_NN_2_next_dtype()
                self._init_tf_vars()

    def _train_epoch_dtype(self, sess, x):
        dtype_nm = dtype2str(self.current_dtype)
        np_dtype = tf2np_dtypes[self.current_dtype]
        x_ = x.astype(np_dtype)

        for i in range(0, int(x_.shape[0]/self.batch_size)):
            x_mb = x_[i*self.batch_size:(i+1)*self.batch_size]
            sess.run(self.train_step[dtype_nm], feed_dict={self.x[dtype_nm]: x_mb})

    def _cast_params_2_next_dtype(self):
        ix = self.dtypes.index(self.current_dtype)
        current_dtype_nm = dtype2str(self.current_dtype)
        next_dtype = self.dtypes[ix + 1]
        next_dtype_nm = dtype2str(next_dtype)

        # Cast current weights to new dtype
        self.weights[next_dtype_nm] = cast_params(self.weights[current_dtype_nm], next_dtype)
        self.biases[next_dtype_nm] = cast_params(self.biases[current_dtype_nm], next_dtype)

    def _update_current_dtype(self):
        ix = self.dtypes.index(self.current_dtype)
        self.current_dtype = self.dtypes[ix + 1]

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
        print('predicting for {}'.format(self.current_dtype))
        dtype_nm = dtype2str(self.current_dtype)

        np_dtype = tf2np_dtypes[self.current_dtype]
        nn = self.NN[dtype_nm]

        return nn.predict(x.astype(np_dtype), batch_size, training)

    def __call__(self, x, training=True):
        dtype_nm = dtype2str(self.current_dtype)

        np_dtype = tf2np_dtypes[self.current_dtype]
        nn = self.NN[dtype_nm]

        return nn(x.astype(np_dtype))
