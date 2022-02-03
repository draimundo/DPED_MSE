#################################
# RAW-to-RGB Model architecture #
#################################

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def dped_g(input_image, leaky = True, norm = 'instance', flat = 0, mix_input=False, onebyone=False, upscale='transpose'):
    with tf.compat.v1.variable_scope("generator"):
        if flat > 0:
            if mix_input:
                flat_in = _extract_colors(input_image)
                flat_conv = _conv_layer(input_image, 60, flat, 2, norm = 'none', leaky = leaky)
                conv1 = _stack(flat_in, flat_conv)
                if onebyone:
                    conv1 = _conv_layer(conv1, 64, 1, 1, norm = 'none', leaky = leaky)
            else:
                conv1 = _conv_layer(input_image, 64, flat, 2, norm = 'none', leaky = leaky)
                if onebyone:
                    conv1 = _conv_layer(conv1, 64, 1, 1, norm = 'none', leaky = leaky)
        else:
            conv1 = _conv_layer(input_image, 64, 9, 1, norm = 'none', leaky = leaky)
            if onebyone:
                conv1 = _conv_layer(conv1, 64, 1, 1, norm = 'none', leaky = leaky)

        conv_b1a = _conv_layer(conv1, 64, 3, 1, norm = norm, leaky = leaky)
        conv_b1b = _conv_layer(conv_b1a, 64, 3, 1, norm = norm, leaky = leaky) + conv1

        conv_b2a = _conv_layer(conv_b1b, 64, 3, 1, norm = norm, leaky = leaky)
        conv_b2b = _conv_layer(conv_b2a, 64, 3, 1, norm = norm, leaky = leaky) + conv_b1b

        conv_b3a = _conv_layer(conv_b2b, 64, 3, 1, norm = norm, leaky = leaky)
        conv_b3b = _conv_layer(conv_b3a, 64, 3, 1, norm = norm, leaky = leaky) + conv_b2b
        

        conv_b4a = _conv_layer(conv_b3b, 64, 3, 1, norm = norm, leaky = leaky)
        conv_b4b = _conv_layer(conv_b4a, 64, 3, 1, norm = norm, leaky = leaky) + conv_b3b


        conv2 = _conv_layer(conv_b4b, 64, 3, 1, norm = 'none', leaky = leaky)
        conv3 = _conv_layer(conv2, 64, 3, 1, norm = 'none', leaky = leaky)
        tconv1 = _upscale(conv3, 64, 3, 2, upscale)
        enhanced = tf.nn.tanh(_conv_layer(tconv1, 3, 9, 1, relu = False, norm = 'none')) * 0.58 + 0.5

    return enhanced

def resnext_g(input_image, leaky = True, norm = 'instance', flat = 0, mix_input=False, onebyone=False, upscale='transpose'):
    with tf.compat.v1.variable_scope("generator"):
        conv1 = _conv_layer(input_image, 64, 4, 2, norm = 'none', leaky = leaky)
        conv1 = _conv_layer(conv1, 64, 1, 1, norm = 'none', leaky = leaky)

        conv_b1 = _convnext(conv1)

        conv_b2 = _convnext(conv_b1)

        conv_b3 = _convnext(conv_b2)

        conv_b4 = _convnext(conv_b3)

        conv2 = _conv_layer(conv_b4, 64, 3, 1, norm = 'none', leaky = leaky)
        conv3 = _conv_layer(conv2, 64, 3, 1, norm = 'none', leaky = leaky)
        tconv1 = _upscale(conv3, 64, 3, 2, upscale)
        enhanced = tf.nn.tanh(_conv_layer(tconv1, 3, 9, 1, relu = False, norm = 'none')) * 0.58 + 0.5
    return enhanced

def swinir_g(input_image, leaky = True, norm = 'instance', flat = 0, mix_input=False, onebyone=False, upscale='resnet'):
    with tf.compat.v1.variable_scope("generator"):
        conv1 = _conv_layer(input_image, 64, 4, 2, norm = 'none', leaky = leaky)
        conv1 = _conv_layer(conv1, 64, 1, 1, norm = 'none', leaky = leaky)

        # conv_b1a = _conv_layer(conv1, 64, 3, 1, norm = norm, leaky = leaky)
        # conv_b1b = _conv_layer(conv_b1a, 64, 3, 1, norm = norm, leaky = leaky) + conv1

        # conv_b2a = _conv_layer(conv_b1b, 64, 3, 1, norm = norm, leaky = leaky)
        # conv_b2b = _conv_layer(conv_b2a, 64, 3, 1, norm = norm, leaky = leaky) + conv_b1b

        conv_b3 = _forward_features(conv_1, dim=64, depth=4, num_heads=4, window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=False, patch_size=1, num_rstb=3)
        
        # conv_b4a = _conv_layer(conv_b3, 64, 3, 1, norm = norm, leaky = leaky)
        # conv_b4b = _conv_layer(conv_b4a, 64, 3, 1, norm = norm, leaky = leaky) + conv_b3

        conv2 = _conv_layer(conv_b3, 64, 3, 1, norm = 'none', leaky = leaky)
        conv3 = _conv_layer(conv2, 64, 3, 1, norm = 'none', leaky = leaky)
        tconv1 = _upscale(conv3, 64, 3, 2, upscale)
        enhanced = tf.nn.tanh(_conv_layer(tconv1, 3, 9, 1, relu = False, norm = 'none')) * 0.58 + 0.5

    return enhanced

def _convnext(input):
    _, rows, cols, in_channels = [i for i in input.get_shape()]

    net = _depth_conv_layer(input, 1, 3 ,1)
    net = _instance_norm(net) # layer norm
    net = _conv_layer(net, in_channels*3, 1, 1, relu=False)
    net = _gelu(net)
    net = _conv_layer(net, in_channels, 1, 1, relu=False)

    return net + input

def _gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def _upscale(net, num_filters, filter_size, factor, method):
    if method == "transpose":
        return _conv_tranpose_layer(net, num_filters, filter_size, factor)
    elif method == "shuffle":
        return _conv_pixel_shuffle_up(net, num_filters, filter_size, factor)
    elif method == "dcl":
        return _pixel_dcl(net, num_filters, filter_size)
    elif method == "resnet":
        return _resblock_up(net, num_filters, sn=True)
    elif method == "nn":
        return _nearest_neighbor(net, factor)
    else:
        print("Unrecognized upscaling method")


def texture_d(image_, activation=True):

    with tf.compat.v1.variable_scope("texture_d"):

        conv1 = _conv_layer(image_, 48, 11, 4, norm = 'none')
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

def unet_d(input, activation=True, norm='instance'):
    with tf.compat.v1.variable_scope("unet_d"):
        ch = 64
        sn = True

        x = _resblock_down(input, ch, use_bias = False, sn=sn, norm=norm) #128
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn, norm=norm) #64

        x = _self_attention(x, ch, sn)
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn, norm=norm) #32
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn, norm=norm) #16
        x = _resblock_down(x, ch, use_bias = False, sn=sn, norm=norm) #8

        x = tf.compat.v1.nn.leaky_relu(x)
        x = _global_sum_pool(x)

        flat = tf.compat.v1.layers.flatten(x)
        out = _fully_connected_layer(flat, 2, relu=False)

        if activation:
            out = tf.nn.softmax(out)
        return out

def _resblock_down(input, num_filters, use_bias=True, sn=False, norm='instance'):
    x = _switch_norm(input, norm)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    x = _switch_norm(x, norm)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 1, relu=False, use_bias=use_bias, sn=sn)

    input = _conv_layer(input, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    return x + input

def _resblock_up(input, num_filters, use_bias=True, sn=False, norm='instance'):
    x = _switch_norm(input, norm)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_tranpose_layer(x, num_filters, 3, 2, relu = False, use_bias = use_bias, sn=sn)

    x = _switch_norm(x, norm)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_tranpose_layer(x, num_filters, 3, 1, relu = False, use_bias = use_bias, sn=sn)

    input = _conv_tranpose_layer(input, num_filters, 3, 2, relu = False, use_bias = use_bias, sn=sn)

    return x + input

def _self_attention(x, num_filters, sn=False):
    f = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    g = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    h = _conv_layer(x, num_filters=num_filters, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)

    s = tf.matmul(_hw_flatten(g), _hw_flatten(f), transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, _hw_flatten(h))
    gamma = tf.Variable(tf.zeros([1]))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    x = gamma * o + x

    return x

def _self_attention_v2(x, num_filters, sn=False):
    f = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    f = _max_pool(f, 2)

    g = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)

    h = _conv_layer(x, num_filters=num_filters, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    h = _max_pool(g, 2)

    s = tf.matmul(_hw_flatten(g), _hw_flatten(f), transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, _hw_flatten(h))
    gamma = tf.Variable(tf.zeros([1]))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    o = _conv_layer(o, num_filters, 1, 1, relu=False, sn=sn)
    x = gamma * o + x

    return x

def _hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def _batch_norm(x):
    return tf.compat.v1.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05)

def weight_variable(shape, name):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def _extract_colors(input):
    return tf.nn.space_to_depth(input, 2)

def _stack(x, y):
    return tf.concat([x, y], 3)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def _conv_multi_block(input, max_size, num_maps, norm):

    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, norm=norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, norm=norm)

    output_tensor = conv_3b

    if max_size >= 5:

        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, norm=norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, norm=norm)

        output_tensor = _stack(output_tensor, conv_5b)

    if max_size >= 7:

        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, norm=norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, norm=norm)

        output_tensor = _stack(output_tensor, conv_7b)

    if max_size >= 9:

        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, norm=norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, norm=norm)

        output_tensor = _stack(output_tensor, conv_9b)

    return output_tensor

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _depth_conv_layer(net, multiplier, filter_size, strides, padding='SAME'):
    _, rows, cols, in_channels = [i for i in net.get_shape()]

    weights_init = _conv_init_vars(net, multiplier, filter_size)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.depthwise_conv2d(net, weights_init, strides_shape, padding=padding)
    return net

def _conv_layer(net, num_filters, filter_size, strides, relu=True, norm='none', padding='SAME', leaky = True, use_bias=True, sn=False):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)

    net = _switch_norm(net, norm)

    if relu:
        if leaky:
            net = tf.compat.v1.nn.leaky_relu(net)
        else:
            net = tf.compat.v1.nn.relu(net)

    return net

def _conv_pixel_shuffle_up(net, num_filters, filter_size, factor):
    weights_init = _conv_init_vars(net, num_filters*factor*2, filter_size)

    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    net = tf.nn.depth_to_space(net, factor)

    return tf.compat.v1.nn.leaky_relu(net)


def _pixel_dcl(net, num_filters, filter_size, d_format='NHWC'):
    axis = (d_format.index('H'), d_format.index('W'))

    conv_0 = _conv_layer(net, num_filters, filter_size, 1, relu=False)
    conv_1 = _conv_layer(conv_0, num_filters, filter_size, 1, relu=False)

    dil_conv_0 = _dilate(conv_0, axis, (0,0))
    dil_conv_1 = _dilate(conv_1, axis, (1,1))

    conv_a = tf.add(dil_conv_0, dil_conv_1)

    weights = _conv_init_vars(conv_a, num_filters, filter_size)
    weights = tf.multiply(weights, _get_mask([filter_size, filter_size, num_filters, num_filters]))
    conv_b =  tf.nn.conv2d(conv_a, weights, strides=[1,1,1,1], padding='SAME')

    out = tf.add(conv_a, conv_b)

    return tf.compat.v1.nn.leaky_relu(out)

def _dilate(net, axes, shifts):
    for index, axis in enumerate(axes):
        elements = tf.unstack(net, axis=axis)
        zeros = tf.zeros_like(elements[0])
        for element_index in range(len(elements), 0, -1):
            elements.insert(element_index-shifts[index], zeros)
        net = tf.stack(elements, axis=axis)
    return net

def _get_mask(shape):
    new_shape = (np.prod(shape[:-2]), shape[-2], shape[-1])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32)

def _switch_norm(net, norm):
    if norm == 'instance':
        return _instance_norm(net)
    elif norm == 'group':
        return _group_norm(net)
    elif norm == 'layer':
        return _layer_norm(net)
    elif norm == 'none':
        return net
    else:
        print("Norm not recognized, using none")
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

def _group_norm(x, G=32, eps=1e-5) :
    N, H, W, C = [i for i in x.get_shape()]
    G = min(G, C)

    x = tf.reshape(x, [N, H, W, G, C // G])
    mean, var = tf.compat.v1.nn.moments(x, [1, 2, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    gamma = tf.Variable(tf.constant(1.0, shape = [1, 1, 1, C]))
    beta = tf.Variable(tf.constant(0.0, shape = [1, 1, 1, C]))

    x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x

def _layer_norm(net):
    if len(net.get_shape()) == 4:
        batch, rows, cols, channels = [i for i in net.get_shape()]
        axes = [1,2,3]
        perm = [3,1,2,0]
    elif len(net.get_shape()) == 3:
        batch, vals, channels = [i for i in net.get_shape()]
        axes = [1,2]
        perm = [2,1,0]
    var_shape = [batch]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, axes, keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    ret = scale*tf.transpose(normalized, perm=perm) + shift
    ret = tf.transpose(ret, perm=perm)

    return ret


def _fully_connected_layer(net, num_weights, bias = True, relu=True):
    channels = net.get_shape().as_list()[-1]
    weights_shape = [channels, num_weights]

    weights = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    out = tf.matmul(net, weights)

    if bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_weights]))
        out = out + bias

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


def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True, leaky = True, use_bias=True, sn=False):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)
    if relu:
        if leaky:
            net = tf.compat.v1.nn.leaky_relu(net)
        else:
            net = tf.compat.v1.nn.relu(net)
    
    return net

def _nearest_neighbor(net, factor=2):
    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_size = [rows * factor, cols * factor]
    return tf.compat.v1.image.resize_nearest_neighbor(net, new_size)

def _max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def _global_sum_pool(x):
    return tf.reduce_sum(x, axis=[1, 2])

def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.Variable(tf.random.normal([1, w_shape[-1]]), dtype=tf.float32)

    u_hat = u
    v_hat = None
    for i in range(iteration):
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


def _patch_embed(net, patch_size, embed_dim, norm_layer):
    B, H, W, C = net.get_shape().as_list()
    if norm_layer:
        net = _layer_norm(net)
    net = tf.reshape(net, [-1, (H//patch_size) * (W//patch_size), embed_dim])
    return net

def _patch_unembed(net, img_size, img_ch):
    B, HW, C = net.get_shape().as_list()
    net = tf.reshape(net, [-1, img_size[0], img_size[1], img_ch])
    return net

def _window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows

def _window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [B, H, W, -1])
    return x

def _window_attention(net, window_size, num_heads, qkv_bias, qk_scale, mask=None):
    B, N, C = [i for i in net.get_shape()]
    head_dim = C // num_heads
    scale = qk_scale or head_dim ** 0.5

    weights_qkv = tf.Variable(tf.random.truncated_normal([C, C*3], stddev=0.02, seed=1), dtype=tf.float32)
    qkv = tf.matmul(net, weights_qkv)
    if qkv_bias:
        bias_qkv = tf.Variable(tf.constant(0.01, shape=[C*3]))
        qkv = qkv + bias_qkv
    qkv_reshape = tf.reshape(qkv, [B, N, 3, num_heads, C//num_heads])
    qkv_reshape = tf.transpose(qkv_reshape, perm=[2,0,3,1,4])
    q, k, v = tf.unstack(qkv_reshape)
    q = q*scale
    attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))

    relative_position_bias_table = tf.Variable(tf.random.truncated_normal([(2*window_size -1) * (2*window_size -1), num_heads], stddev=0.02, seed=1), dtype=tf.float32)
    coords_h = tf.range(window_size)
    coords_w = tf.range(window_size)
    coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
    coords_flatten = tf.reshape(coords, [2, -1])  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = tf.transpose(relative_coords, perm=[1,2,0]) # Wh*Ww, Wh*Ww, 2
    relative_coords = relative_coords + [window_size - 1, window_size - 1]  # shift to start from 0
    relative_coords = relative_coords * [2*window_size - 1, 1]
    relative_position_index = tf.math.reduce_sum(relative_coords,-1)  # Wh*Ww, Wh*Ww

    relative_position_bias = tf.reshape(tf.gather(relative_position_bias_table, tf.reshape(relative_position_index, [-1])), [window_size * window_size, window_size * window_size, -1])  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
        nW = mask.get_shape()[0]  # tf.shape(mask)[0]
        attn = tf.reshape(attn, shape=[-1, nW, num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
        attn = tf.reshape(attn, shape=[-1, num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
    else:
        attn = tf.nn.softmax(attn, axis=-1)

    x = tf.reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]), [B, N, C])

    weights_proj = tf.Variable(tf.random.truncated_normal([C, C], stddev=0.02, seed=1), dtype=tf.float32)
    proj = tf.matmul(x, weights_proj)
    bias_proj = tf.Variable(tf.constant(0.01, shape=[C]))
    proj = proj + bias_proj

    return x

def _forward_features(input, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, patch_size, num_rstb):
    B, H, W, C = input.get_shape().as_list()
    x_size = (H, W)
    input_resolution = x_size

    net = _patch_embed(input, patch_size, dim, True)
    
    for i in range(num_rstb):
        net = _rstb(net, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, patch_size, C, x_size)
    net = _layer_norm(net)
    net = _patch_unembed(net, x_size, C)
    return net

def _rstb(input, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, patch_size, img_ch, x_size):
    net = _swin_transformer_layer(input, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, x_size)
    net = _patch_unembed(net, x_size, img_ch)
    net = _conv_layer(net, dim//2, 3, 1, relu=False)
    net = _patch_embed(net, patch_size, dim, False) + input
    return net

def _swin_transformer_layer(net, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, x_size):
    B, HW, C = net.get_shape().as_list()
    for i in range(depth):
        net = _swin_transformer_block(net, dim, num_heads, window_size,
                                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, x_size=x_size)
    return net

def _swin_transformer_block(x, dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, x_size):
    B, L, C = x.get_shape().as_list()
    H, W = x_size

    shortcut = x
    x = _layer_norm(x)
    x = tf.reshape(x, shape=[-1, H, W, C])

    if shift_size > 0:
        shifted_x = tf.roll(x, shift=[-shift_size, -shift_size], axis=[1, 2])
    else:
        shifted_x = x

    x_windows = _window_partition(shifted_x, window_size)
    x_windows = tf.reshape(x_windows, shape=[-1, window_size * window_size, C])

    if shift_size > 0:
        attn_mask = _attention_mask(x_size, window_size, shift_size)
    else:
        attn_mask = None
    attn_windows = _window_attention(x_windows, window_size, num_heads, qkv_bias, qk_scale, attn_mask)

    attn_windows = tf.reshape(attn_windows, [-1, window_size, window_size, C])
    shifted_x = _window_reverse(attn_windows, window_size, H, W)  # B H' W' C

    if shift_size > 0:
        x = tf.roll(shifted_x, shift=[shift_size, shift_size], axis=(1, 2))
    else:
        x = shifted_x
    x = tf.reshape(x, [B, H * W, C])

    x = shortcut + x #todo add drop_path
    x = x + _mlp(_layer_norm(x), hidden_features=dim*mlp_ratio)
    return x

def _attention_mask(input_resolution, window_size, shift_size):
    H, W = input_resolution
    img_mask = np.zeros([1, H, W, 1])  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    img_mask = tf.constant(img_mask)
    mask_windows = _window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = tf.reshape(mask_windows, [-1, window_size * window_size])
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attn_mask = tf.where(tf.abs(attn_mask)<0.5, tf.zeros_like(attn_mask), tf.ones_like(attn_mask)*-100.)
    return attn_mask

def _mlp(net, hidden_features, out_features=None):
    B, L, C = net.get_shape().as_list()
    
    out_features = out_features or C
    hidden_features = hidden_features or C

    net = _fully_connected_layer(net, hidden_features, bias=True, relu=False)
    net = _gelu(net)
    #todo drop?
    net = _fully_connected_layer(net, out_features, bias=True, relu=False)
    #todo drop?
    return net


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = np.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    x = np.transpose(x, axes=[0, 1, 3, 2, 4, 5])
    windows = np.reshape(x, [-1, window_size, window_size, C])
    return windows

if __name__ == "__main__":
    window_size = 8
    shift_size = 4
    input_resolution = (16, 16)
    # img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
    # h_slices = (slice(0, -window_size),
    #             slice(-window_size, -shift_size),
    #             slice(-shift_size, None))
    # w_slices = (slice(0, -window_size),
    #             slice(-window_size, -shift_size),
    #             slice(-shift_size, None))
    # cnt = 0
    # for h in h_slices:
    #     for w in w_slices:
    #         img_mask[:, h, w, :] = cnt
    #         cnt += 1

    # mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    # mask_windows = np.reshape(mask_windows,[-1, window_size * window_size])

    # attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    # attn_mask = np.where(attn_mask==0, 0., -100.)

    with tf.compat.v1.Session() as sess:
        attn_mask = sess.run(_attention_mask(input_resolution, window_size, shift_size))


    # plt.matshow(img_mask[0, :, :, 0])
    plt.matshow(attn_mask[0])
    plt.matshow(attn_mask[1])
    plt.matshow(attn_mask[2])
    plt.matshow(attn_mask[3])

    plt.show()


