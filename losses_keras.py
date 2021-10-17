import tensorflow as tf
import numpy as np
import vgg
import scipy.stats as st
from skimage.filters import window

def loss_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def loss_fourier(y_true, y_pred):
    hann2d = window('hann', (256, 256))
    #TODO process by color (mult by win + rfft2d)
    #TODO compute mag and phase


class loss_content(tf.keras.losses.Loss):
    def __init__(self, vgg_dir, name="loss_content"):
        super().__init__(name=name)
        self.vgg_dir = vgg_dir

    def call(self, y_true, y_pred):
        CONTENT_LAYER = 'relu5_4'
        y_true = vgg.net(self.vgg_dir, vgg.preprocess(y_true * 255))
        y_pred = vgg.net(self.vgg_dir, vgg.preprocess(y_pred * 255))
        return tf.reduce_mean(tf.math.squared_difference(y_true[CONTENT_LAYER], y_pred[CONTENT_LAYER]))

def loss_color(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(_blur(y_true), _blur(y_pred)))

def loss_psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))

def loss_ssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def _gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def _blur(x):
    kernel_var = _gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')