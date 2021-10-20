import tensorflow as tf
import numpy as np
import vgg
import scipy.stats as st
from skimage.filters import window


class loss_creator(tf.keras.losses.Loss):
    def __init__(self, vgg_dir, patch_w, patch_h, mse, fourier, content, color, ssim, name="custom_loss"):
        super().__init__(name=name)
        self.facMse = mse
        self.facFourier = fourier
        if self.facFourier > 0:
            self.fourier = loss_fourier(patch_w, patch_h)
        self.facContent = content
        if self.facContent > 0:
            self.content = loss_content(vgg_dir)
        self.facColor = color
        self.facSsim = ssim
    def call(self, y_true, y_pred):
        loss = 0.0
        if self.facMse > 0:
            loss += self.facMse * loss_mse(y_true, y_pred)
        if self.facFourier > 0:
            loss += self.facFourier * self.fourier(y_true, y_pred)
        if self.facContent > 0:
            loss += self.facContent * self.content(y_true, y_pred)
        if self.facColor > 0:
            loss += self.facColor * loss_color(y_true, y_pred)
        if self.facSsim > 0:
            loss += self.facSsim * loss_ssim(y_true, y_pred)
        return loss

def loss_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

class loss_fourier(tf.keras.losses.Loss):
    def __init__(self, patch_w, patch_h, name="loss_fourier"):
        super().__init__(name=name)
        h2d = np.float32(window('hann', (patch_w, patch_h)))
        self.hann2d = tf.stack([h2d,h2d,h2d],axis=2) #stack for 3 color channels
        self.patch_w = patch_w
        self.patch_h = patch_h
    
    def call(self, y_true, y_pred):
        y_true = tf.multiply(y_true, self.hann2d) #filter with hann window
        y_pred = tf.multiply(y_pred, self.hann2d)

        err_mag = 0.0
        err_ang = 0.0

        for ch in range(y_true.shape[-1]): #iterate through channels
            true_mag = tf.abs(y_true[..., ch])
            true_ang = tf.math.angle(y_true[..., ch])
            pred_mag = tf.abs(y_pred[..., ch])
            pred_ang = tf.math.angle(y_pred[..., ch])

            err_mag += tf.reduce_mean(tf.abs(true_mag - pred_mag))
            err_ang += tf.reduce_mean(tf.abs(true_ang - pred_ang))

        return (err_mag + err_ang)/2

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

def metr_psnr(y_true, y_pred):
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