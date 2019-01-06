# MultidtypeNN
Progressively increasing datatype precision during neural network training

## Example
----------------
Initialize tensorflow session and multitypeNN:
```
sess = tf.Session()
mdtnn = MultidtypeNN(n_nodes=[5, 64, 64, 64, 1],
                     activation_list=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
                     weight_init_method=None,
                     bias_init_method=None,
                     trainable=True,
                     with_bn=False,
                     dropout_rate=0,
                     tb_flag=False,
                     dtypes=[tf.float16, tf.float32])
```

Initialize multitypeNN object with the training hyper parameters:
```
mdtnn.train_init(dim_input=5,
                 y=y,
                 learning_rate=0.3,
                 opt_nm='sgd',
                 cost_nm='mse',
                 batch_size=32,
                 n_epochs_per_precision=200)
```

Train neural network gradually increasing the precision every 200 epochs:
```
mdtnn.train(sess, x)
```

Generate predictions using the highest precision network:
```
preds = mdtnn.predict(x, batch_size=32)
print(sess.run(preds))
```
