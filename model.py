#################################
# RAW-to-RGB Model architecture #
#################################

import tensorflow as tf
import numpy as np


def resnet(input_image):

    with tf.compat.v1.variable_scope("generator"):

        W1 = weight_variable([9, 9, 4, 64], name="W1"); b1 = bias_variable([64], name="b1");
        c1 = leaky_relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2");
        c2 = leaky_relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3");
        c3 = leaky_relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4");
        c4 = leaky_relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5");
        c5 = leaky_relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6"); b6 = bias_variable([64], name="b6");
        c6 = leaky_relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7"); b7 = bias_variable([64], name="b7");
        c7 = leaky_relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8"); b8 = bias_variable([64], name="b8");
        c8 = leaky_relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9"); b9 = bias_variable([64], name="b9");
        c9 = leaky_relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10"); b10 = bias_variable([64], name="b10");
        c10 = leaky_relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11"); b11 = bias_variable([64], name="b11");
        c11 = leaky_relu(conv2d(c10, W11) + b11)

        # Transposed convolution

        c12 = _conv_tranpose_layer(c11, 64, 3, 2)

        # Final

        W13 = weight_variable([9, 9, 64, 3], name="W13"); b13 = bias_variable([3], name="b13");
        enhanced = tf.nn.tanh(conv2d(c12, W13) + b13) * 0.58 + 0.5

    return enhanced


def PUNET(input, instance_norm=False, instance_norm_level_1=False, num_maps_base=16):

    with tf.compat.v1.variable_scope("generator"):

        # -----------------------------------------
        # Downsampling layers
        conv_l1_d1 = _conv_multi_block(input, 3, num_maps=num_maps_base, instance_norm=False)              # 128 -> 128
        pool1 = max_pool(conv_l1_d1, 2)                                                         # 128 -> 64

        conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=num_maps_base*2, instance_norm=instance_norm)      # 64 -> 64
        pool2 = max_pool(conv_l2_d1, 2)                                                         # 64 -> 32

        conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=num_maps_base*4, instance_norm=instance_norm)     # 32 -> 32
        pool3 = max_pool(conv_l3_d1, 2)                                                         # 32 -> 16

        conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=num_maps_base*8, instance_norm=instance_norm)     # 16 -> 16
        pool4 = max_pool(conv_l4_d1, 2)                                                         # 16 -> 8

        # -----------------------------------------
        # Processing: Level 5,  Input size: 8 x 8
        conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=num_maps_base*16, instance_norm=instance_norm)
        conv_l5_d2 = _conv_multi_block(conv_l5_d1, 3, num_maps=num_maps_base*16, instance_norm=instance_norm) + conv_l5_d1
        conv_l5_d3 = _conv_multi_block(conv_l5_d2, 3, num_maps=num_maps_base*16, instance_norm=instance_norm) + conv_l5_d2
        conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=num_maps_base*16, instance_norm=instance_norm)

        conv_t4b = _conv_tranpose_layer(conv_l5_d4, num_maps_base*8, 3, 2)      # 8 -> 16

        # -----------------------------------------
        # Processing: Level 4,  Input size: 16 x 16
        conv_l4_d6 = conv_l4_d1
        conv_l4_d7 = stack(conv_l4_d6, conv_t4b)
        conv_l4_d8 = _conv_multi_block(conv_l4_d7, 3, num_maps=num_maps_base*8, instance_norm=instance_norm)

        conv_t3b = _conv_tranpose_layer(conv_l4_d8, num_maps_base*4, 3, 2)      # 16 -> 32

        # -----------------------------------------
        # Processing: Level 3,  Input size: 32 x 32
        conv_l3_d6 = conv_l3_d1
        conv_l3_d7 = stack(conv_l3_d6, conv_t3b)
        conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=num_maps_base*4, instance_norm=instance_norm)

        conv_t2b = _conv_tranpose_layer(conv_l3_d8, num_maps_base*2, 3, 2)       # 32 -> 64

        # -------------------------------------------
        # Processing: Level 2,  Input size: 64 x 64
        conv_l2_d7 = conv_l2_d1
        conv_l2_d8 = stack(_conv_multi_block(conv_l2_d7, 3, num_maps=num_maps_base*2, instance_norm=instance_norm), conv_t2b)
        conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=num_maps_base*2, instance_norm=instance_norm)

        conv_t1b = _conv_tranpose_layer(conv_l2_d9, num_maps_base, 3, 2)       # 64 -> 128

        # -------------------------------------------
        # Processing: Level 1,  Input size: 128 x 128
        conv_l1_d9 = conv_l1_d1
        conv_l1_d10 = stack(_conv_multi_block(conv_l1_d9, 3, num_maps=num_maps_base, instance_norm=False), conv_t1b)
        conv_l1_d11 = stack(conv_l1_d10, conv_l1_d1)
        conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=num_maps_base, instance_norm=False)

        # ----------------------------------------------------------
        # Processing: Level 0 (x2 upscaling),  Input size: 128 x 128
        conv_l0 = _conv_tranpose_layer(conv_l1_d12, num_maps_base//4, 3, 2)        # 128 -> 256
        conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)

        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5
        
    output_l0 = tf.identity(output_l0, name='output_l0')

    return output_l0


def weight_variable(shape, name):

    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_multi_block(input, max_size, num_maps, instance_norm):

    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:

        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_5b)

    if max_size >= 7:

        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_7b)

    if max_size >= 9:

        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_9b)

    return output_tensor


def stack(x, y):
    return tf.concat([x, y], 3)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding) + bias

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = tf.compat.v1.nn.leaky_relu(net)

    return net


def _instance_norm(net):

    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, [1,2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.compat.v1.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    return tf.compat.v1.nn.leaky_relu(net)


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')
