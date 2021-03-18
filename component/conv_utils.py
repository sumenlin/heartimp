import numpy as np
import tensorflow as tf

def encode_pconv_block(conv_output, mask, kernel, layer_id, n_output, activation, regularizer):
    update_mask = tf.layers.conv1d(mask,
                                   filters=1,
                                   kernel_size=kernel,
                                   strides=2,
                                   padding='same',
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=tf.constant_initializer(1.0),
                                   bias_initializer=tf.constant_initializer(0.0),
                                   trainable=False,
                                   name='layer1_mask_%s' % str(layer_id))
    mask_ratio = kernel / (update_mask + 1e-8)
    update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
    mask_ratio = mask_ratio * update_mask
    print('update_mask', update_mask.get_shape())

    conv_output1 = tf.layers.conv1d(conv_output,
                                    filters=n_output,
                                    kernel_size=kernel,
                                    strides=2,
                                    padding='same',
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=regularizer,
                                    trainable=True,
                                    name='layer1_%s' % layer_id)

    conv_output = activation(conv_output1 * mask_ratio)  #

    # conv_output = tf.layers.max_pooling1d(conv_output,
    #                                       pool_size=3,
    #                                       strides=2,
    #                                       padding='valid',
    #                                       name='layer_%s_maxpool' % layer_id)
    return conv_output, update_mask


def decode_pconv_block(input, input_mask, lateral_input, lateral_input_mask, kernel_size, depth_output, stride,activation, name, regularizer):
    with tf.variable_scope(name):
        upsample_input = tf.keras.layers.UpSampling1D(size=stride)(input)
        upsample_mask = tf.keras.layers.UpSampling1D(size=stride)(input_mask)


        print('upsample_input', upsample_input.get_shape(), 'lateral_input', lateral_input.get_shape())

        update_mask1 = tf.layers.conv1d(upsample_mask,
                                       filters=1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding='same',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.constant_initializer(1.0),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       trainable=False,
                                       name='layer1_mask_decode1_%s' % name)
        update_mask2 = tf.layers.conv1d(lateral_input_mask,
                                       filters=1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding='same',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.constant_initializer(1.0),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       trainable=False,
                                       name='layer1_mask_decode2_%s' % name)
        upsample_input_channel = tf.cast(tf.shape(upsample_input)[-1], tf.float32)
        lateral_input_channel = tf.cast(tf.shape(lateral_input)[-1], tf.float32)
        update_mask = update_mask1 * (upsample_input_channel/(upsample_input_channel+lateral_input_channel))+ \
                      update_mask2 * (lateral_input_channel/(upsample_input_channel+lateral_input_channel))
        mask_ratio = kernel_size/ (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask


        conv_output = tf.concat([upsample_input, lateral_input], axis=-1)
        conv_output1 = tf.layers.conv1d(conv_output,
                                        filters=depth_output,
                                        kernel_size=kernel_size,
                                        strides=1,
                                        padding='same',
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=regularizer,
                                        trainable=True,
                                        name='layer1_decode_%s' % name)

        output = activation(conv_output1 * mask_ratio)
    return output, update_mask



def encode_3d_pconv_block(conv_output, mask, kernel, stride_size, dilation_rate, layer_id, n_output, regularizer, activation = tf.nn.elu):
    if isinstance(kernel,int):
        if dilation_rate ==1:
            kernel_2d = 1
        else:
            kernel_2d = 3
    else:
        kernel, kernel_2d = kernel

    update_mask = tf.layers.conv2d(mask,
                                   filters=1,
                                   kernel_size=(kernel, kernel_2d),
                                   strides=(stride_size, 1),
                                   padding='same',
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=tf.constant_initializer(1.0),
                                   bias_initializer=tf.constant_initializer(0.0),
                                   dilation_rate=(dilation_rate, 1),
                                   trainable=False,
                                   name='layer1_mask_%s' % str(layer_id))
    mask_ratio = kernel*kernel_2d / (update_mask + 1e-8)
    update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
    mask_ratio = mask_ratio * update_mask
    print('update_mask', update_mask.get_shape())

    conv_output1 = tf.layers.conv2d(conv_output,
                                    filters=n_output,
                                    kernel_size=(kernel, kernel_2d),
                                    strides=(stride_size,1),
                                    padding='same',
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=regularizer,
                                    dilation_rate=(dilation_rate, 1),
                                    trainable=True,
                                    name='layer1_encode_%s' % layer_id)
    conv_output = activation(conv_output1 * mask_ratio)
    return conv_output, update_mask

def decode_3d_pconv_block(input, input_mask, lateral_input, lateral_input_mask, kernel, depth_output, stride_size, dilation_rate,activation, name, regularizer):
    if isinstance(kernel,int):
        if dilation_rate ==1:
            kernel_2d = 1
        else:
            kernel_2d = 3
    else:
        kernel, kernel_2d = kernel

    with tf.variable_scope(name):
        upsample_input = tf.keras.layers.UpSampling2D(size=(stride_size, 1))(input)
        upsample_mask = tf.keras.layers.UpSampling2D(size=(stride_size, 1))(input_mask)

        print('upsample_input', upsample_input.get_shape(), 'lateral_input', lateral_input.get_shape())


        update_mask1 = tf.layers.conv2d(upsample_mask,
                                       filters=1,
                                       kernel_size=(kernel, kernel_2d),
                                       strides=(1, 1),
                                       padding='same',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.constant_initializer(1.0),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       dilation_rate=(dilation_rate, 1),
                                       trainable=False,
                                       name='layer1_decode_mask1_%s' % str(name))
        update_mask2 = tf.layers.conv2d(lateral_input_mask,
                                       filters=1,
                                       kernel_size=(kernel, kernel_2d),
                                       strides=(1, 1),
                                       padding='same',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.constant_initializer(1.0),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       dilation_rate=(dilation_rate, 1),
                                       trainable=False,
                                       name='layer1_decode_mask2_%s' % str(name))
        upsample_input_channel = tf.cast(tf.shape(upsample_input)[-1], tf.float32)
        lateral_input_channel = tf.cast(tf.shape(lateral_input)[-1], tf.float32)
        update_mask = update_mask1 * (upsample_input_channel/(upsample_input_channel+lateral_input_channel))+ \
                      update_mask2 * (lateral_input_channel/(upsample_input_channel+lateral_input_channel))
        mask_ratio = kernel*kernel_2d / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        conv_output = tf.concat([upsample_input, lateral_input], axis=-1)
        conv_output1 = tf.layers.conv2d(conv_output,
                                        filters=depth_output,
                                        kernel_size=(kernel, kernel_2d),
                                        strides=(1, 1),
                                        padding='same',
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=regularizer,
                                        dilation_rate=(dilation_rate, 1),
                                        trainable=True,
                                        name='layer1_decode_%s' % name)
        conv_output = activation(conv_output1 * mask_ratio)
    return conv_output, update_mask

def bottle_neck_encode(current_input, bottle_neck_dim, n_mlp, keep_rate, regularizer, reuse=False):
    dim0, dim1, dim2 = current_input.get_shape().as_list()
    hidden_dim = dim1 * dim2
    mlp_layers = list(np.linspace(hidden_dim, bottle_neck_dim, n_mlp + 1, endpoint=True, dtype=int))
    current_input = tf.reshape(current_input, [-1, hidden_dim])
    with tf.variable_scope('bn_encode', reuse=reuse):
        for idx, output_dim in enumerate(mlp_layers[1:]):
            current_input = tf.layers.dense(current_input,
                                            output_dim,
                                            activation=tf.nn.tanh,
                                            trainable=True,
                                            kernel_regularizer=regularizer,
                                            name='neck_down%s' % idx)
            current_input = tf.nn.dropout(current_input, keep_rate)
        return current_input, mlp_layers, dim1, dim2

def bottle_neck_decode(mlp_layers, current_input_, dim1, dim2, keep_rate, regularizer, reuse=False):
    with tf.variable_scope('bn_decode', reuse=reuse):

        for idx, output_dim in enumerate(mlp_layers[1:]):
            current_input_ = tf.layers.dense(current_input_,
                                             output_dim,
                                             activation=tf.nn.tanh,
                                             trainable=True,
                                             kernel_regularizer=regularizer,
                                             name='neck_up%s' % idx)
            current_input_ = tf.nn.dropout(current_input_, keep_rate)
        current_input_ = tf.reshape(current_input_, [-1, dim1, dim2])
        return current_input_


def encode_block(conv_output, kernel, layer_id, n_output, regularizer, stride = 2, dilation_rate = 1, activation = tf.nn.relu):
    conv_output1 = tf.layers.conv1d(conv_output,
                                    filters=n_output,
                                    kernel_size=kernel,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=regularizer,
                                    dilation_rate=dilation_rate,
                                    trainable=True,
                                    name='layer1_%s' % layer_id)

    return conv_output1

def decode_block(input, kernel_size, depth_output, stride, activation, regularizer, name, dilation_rate = 1):

    with tf.variable_scope(name):
        upsample_input = tf.keras.layers.UpSampling1D(size=stride)(input)
        conv_output1 = tf.layers.conv1d(upsample_input,
                                        filters=depth_output,
                                        kernel_size=kernel_size,
                                        strides=1,
                                        padding='same',
                                        activation=activation,
                                        use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=regularizer,
                                        dilation_rate=dilation_rate,
                                        trainable=True,
                                        name='layer1_decode_%s' % name)
    return conv_output1

