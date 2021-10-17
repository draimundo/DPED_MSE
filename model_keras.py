import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def dped_resnet(input_shape, instance_norm = True):
    inputs = keras.Input(shape=input_shape)

    conv_0 = layers.Conv2D(64, 9, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(inputs)

    conv_b1 = _dped_resnet_block(conv_0, instance_norm)
    conv_b2 = _dped_resnet_block(conv_b1, instance_norm)
    conv_b3 = _dped_resnet_block(conv_b2, instance_norm)
    conv_b4 = _dped_resnet_block(conv_b3, instance_norm)

    conv_5 = layers.Conv2D(64, 9, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv_b4)
    conv_5 = tf.nn.leaky_relu(conv_5)

    conv_6 = layers.Conv2D(64, 9, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv_5)
    conv_6 = tf.nn.leaky_relu(conv_6)

    tconv_7 = layers.Conv2DTranspose(64, 3, (2,2), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv_6)
    tconv_7 = tf.nn.leaky_relu(tconv_7)

    conv_8 = layers.Conv2D(3, 9, padding="same", activation=tf.nn.tanh, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(tconv_7) * 0.58 + 0.5

    return Model(inputs=inputs, outputs=conv_8, name="dped_resnet")

def _dped_resnet_block(input, instance_norm):
    x = layers.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(input)
    if instance_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(x)
    if instance_norm:
        x = layers.BatchNormalization()(x)
    return layers.add([input, x])

def dped_adversarial(input_shape):
    inputs = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(48, 11, (4,4), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(inputs)

    conv2 = keras.layers.Conv2D(128, 5, (2,2), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv1)
    conv2 = layers.BatchNormalization()(conv2)

    conv3 = keras.layers.Conv2D(192, 3, (1,1), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv2)
    conv3 = layers.BatchNormalization()(conv3)

    conv4 = keras.layers.Conv2D(192, 3, (1,1), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv3)
    conv4 = layers.BatchNormalization()(conv4)

    conv5 = keras.layers.Conv2D(192, 3, (1,1), padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Flatten()(conv5)

    fc6 = layers.Dense(1024, activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(conv5)
    
    fc7 = layers.Dense(2, activation=tf.nn.softmax, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(fc6)

    return Model(inputs=inputs, outputs=fc7, name="dped_adversarial")

def fourier_adversarial(input_shape):
    inputs = keras.Input(shape=input_shape)
    finputs = layers.Flatten()(inputs)

    fc1 = layers.Dense(1024, activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(finputs)
    
    fc2 = layers.Dense(1024, activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(fc1)

    fc3 = layers.Dense(1024, activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(fc2)

    fc4 = layers.Dense(1024, activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(fc3)

    fc5 = layers.Dense(2, activation=tf.nn.softmax, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(fc4)

    return Model(inputs=inputs, outputs=fc5, name="fourier_adversarial")