import tensorflow as tf


def fully_connected(input_data, units,
                    kernel=None,
                    bias=None,
                    activation=tf.nn.relu,
                    trainable=True,
                    regularizer=None,
                    name=''):
    if kernel == None:
        use_bias = (bias != None)
        return tf.layers.dense(input_data, units, activation=activation,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               kernel_regularizer=regularizer,
                               trainable=trainable,
                               use_bias=use_bias,
                               bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name=name)
    else:
        with tf.variable_scope(name):
            x = tf.matmul(kernel, input_data)

            if bias != None:
                x = tf.nn.bias_add(x, bias)
            x = activation(x)
            return x


def conv2D(input_data,
           kernel_shape,
           kernel=None,
           strides=(1, 1, 1, 1),
           padding='SAME',
           trainable=True,
           activation=tf.nn.relu,
           bias=None,
           name='',
           regularizer=None,
           dropout=0.0):
    with tf.variable_scope(name):
        if kernel == None:
            weight = tf.get_variable("weight", shape=kernel_shape,
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     regularizer=regularizer)
        else:
            weight = kernel

        conv = tf.nn.conv2d(input_data, weight, strides=strides, padding=padding)

        if bias == "bn":
            conv = tf.layers.batch_normalization(conv,
                                                 beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 gamma_regularizer=regularizer,
                                                 beta_regularizer=regularizer,
                                                 training=trainable)
        elif bias == "bias":
            bias = tf.get_variable(name="bias", shape=kernel_shape[-1],
                                   trainable=trainable,
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0),
                                   regularizer=regularizer)

            conv = tf.nn.bias_add(conv, bias)

        if activation != None:
            conv = activation(conv)

        if (dropout < 1 and dropout > 0):
            conv = tf.layers.dropout(conv, rate=dropout)

        return conv


def deconv2D(input_tensor,
             kernel_shape,
             output_shape,
             kernel=None,
             strides=(1, 1, 1, 1),
             padding='SAME',
             trainable=True,
             activate=True,
             bias=None,
             name='',
             regularizer=None,
             dropout=0.0):
    with tf.variable_scope(name):
        if kernel == None:
            weight = tf.get_variable(name="weight",
                                     shape=kernel_shape,
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     regularizer=regularizer)
        else:
            weight = kernel

        deconv = tf.nn.conv2d_transpose(input_tensor, weight, output_shape, strides, padding=padding)

        if bias == "bn":
            deconv = tf.layers.batch_normalization(deconv, beta_initializer=tf.zeros_initializer(),
                                                   gamma_initializer=tf.ones_initializer(),
                                                   moving_mean_initializer=tf.zeros_initializer(),
                                                   moving_variance_initializer=tf.ones_initializer(),
                                                   gamma_regularizer=regularizer,
                                                   beta_regularizer=regularizer,
                                                   training=trainable)
        elif bias == "bias":
            window_height, window_width, num_output_channels, num_input_channels = weight.get_shape()
            bias = tf.get_variable(name="bias",
                                   shape=num_output_channels,
                                   trainable=trainable,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                   regularizer=regularizer)
            deconv = tf.nn.bias_add(deconv, bias)

        if activate == True:
            deconv = tf.nn.relu(deconv)

        if (dropout < 1):
            deconv = tf.layers.dropout(deconv, dropout)

        return deconv


def max_pooling(input_data, pool_size, strides, name=''):
    return tf.layers.max_pooling2d(input_data, pool_size=pool_size, strides=strides, name=name)


def upsample(input_tensor, new_tensor_shape, name=''):
    _, new_h, new_w, _ = new_tensor_shape
    with tf.variable_scope(name):
        upsampled = tf.image.resize_images(input_tensor, (new_h, new_w),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return upsampled


def unfold(input_tensor, name=None):
    with tf.variable_scope(name):
        input_tensor_shape = input_tensor.get_shape().as_list()
        input_tensor_data_n = 1
        for shape in input_tensor_shape[1:]:
            input_tensor_data_n = input_tensor_data_n * shape
        return tf.reshape(input_tensor, [-1, input_tensor_data_n], name=name)


def fold(input_tensor, new_shape, name=None):
    return tf.reshape(input_tensor, new_shape, name=name)
