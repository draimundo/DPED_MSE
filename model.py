#################################
# RAW-to-RGB Model architecture #
#################################

import tensorflow as tf
import numpy as np


def dped_g(input_image, leaky = True, instance_norm = True, flat = 0):
    with tf.compat.v1.variable_scope("generator"):
        if flat > 0:
            conv1 = _conv_layer(input_image, 64, flat, 2, instance_norm = False, leaky = leaky)
        else:
            conv1 = _conv_layer(input_image, 64, 9, 1, instance_norm = False, leaky = leaky)
        
        conv_b1a = _conv_layer(conv1, 64, 3, 1, instance_norm = instance_norm, leaky = leaky)
        conv_b1b = _conv_layer(conv_b1a, 64, 3, 1, instance_norm = instance_norm, leaky = leaky) + conv1

        conv_b2a = _conv_layer(conv_b1b, 64, 3, 1, instance_norm = instance_norm, leaky = leaky)
        conv_b2b = _conv_layer(conv_b2a, 64, 3, 1, instance_norm = instance_norm, leaky = leaky) + conv_b1b

        conv_b3a = _conv_layer(conv_b2b, 64, 3, 1, instance_norm = instance_norm, leaky = leaky)
        conv_b3b = _conv_layer(conv_b3a, 64, 3, 1, instance_norm = instance_norm, leaky = leaky) + conv_b2b

        conv_b4a = _conv_layer(conv_b3b, 64, 3, 1, instance_norm = instance_norm, leaky = leaky)
        conv_b4b = _conv_layer(conv_b4a, 64, 3, 1, instance_norm = instance_norm, leaky = leaky) + conv_b3b

        conv2 = _conv_layer(conv_b4b, 64, 3, 1, instance_norm = False, leaky = leaky)
        conv3 = _conv_layer(conv2, 64, 3, 1, instance_norm = False, leaky = leaky)
        tconv1 = _conv_tranpose_layer(conv3, 64, 3, 2, leaky = leaky)
        enhanced = tf.nn.tanh(_conv_layer(tconv1, 3, 9, 1, relu = False, instance_norm = False)) * 0.58 + 0.5

    return enhanced

def texture_d(image_, activation=True):

    with tf.compat.v1.variable_scope("texture_d"):

        conv1 = _conv_layer(image_, 48, 11, 4, instance_norm = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat = tf.compat.v1.layers.flatten(conv5)

        fc1 = _fully_connected_layer(flat, 1024)
        adv_out = _fully_connected_layer(fc1, 1, relu=False)
        
        if activation:
            adv_out = tf.nn.sigmoid(adv_out)
    
    return adv_out

def fourier_d(input, activation=True):
    with tf.compat.v1.variable_scope("fourier_d"):
        
        flat = tf.compat.v1.layers.flatten(input)

        fc1 = _fully_connected_layer(flat, 1024)
        fc2 = _fully_connected_layer(fc1, 1024)
        fc3 = _fully_connected_layer(fc2, 1024)
        fc4 = _fully_connected_layer(fc3, 1024)

        out = _fully_connected_layer(fc4, 2, relu=False)
        if activation:
            out = tf.nn.softmax(out)

    return out

def unet_d(input, activation=True):
    with tf.compat.v1.variable_scope("unet_d"):
        ch = 64
        sn = True

        x = _resblock_down(input, ch, use_bias = False, sn=sn) #128
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #64

        x = _self_attention(x, ch, sn)
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #32
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #16
        x = _resblock_down(x, ch, use_bias = False, sn=sn) #8

        x = leaky_relu(x)
        x = _global_sum_pool(x)

        flat = tf.compat.v1.layers.flatten(x)
        out = _fully_connected_layer(flat, 2, relu=False)

        if activation:
            out = tf.nn.softmax(out)


def _resblock_down(input, num_filters, use_bias=True, sn=False):
    x = _batch_norm(input)
    x = leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    x = _batch_norm(x)
    x = leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 1, relu=False, use_bias=use_bias, sn=sn)

    input = _conv_layer(input, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    return x + input

def _self_attention(x, num_filters, sn=False):
    f = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    g = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    h = _conv_layer(x, num_filters=num_filters, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)

    s = tf.matmul(_hw_flatten(g), _hw_flatten(f), transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, _hw_flatten(h))
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    x = gamma * o + x

    return x

def _hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def _batch_norm(x):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05)

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


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME', leaky = True, use_bias=True, sn=False):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        if leaky:
            net = tf.compat.v1.nn.leaky_relu(net)
        else:
            net = tf.compat.v1.nn.relu(net)

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

def _fully_connected_layer(net, num_weights, relu=True):
    batch, channels = [i for i in net.get_shape()]
    weights_shape = [channels, num_weights]

    weights = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.01, shape=[num_weights]))

    out = tf.matmul(net, weights) + bias

    if relu:
        out = tf.compat.v1.nn.leaky_relu(out)

    return out



def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.compat.v1.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _conv_tranpose_layer(net, num_filters, filter_size, strides, leaky = True):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    if leaky:
        return tf.compat.v1.nn.leaky_relu(net)
    else:
        return tf.compat.v1.nn.relu(net)


def _max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def _global_sum_pool(x):
    return tf.reduce_sum(x, axis=[1, 2])

def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
