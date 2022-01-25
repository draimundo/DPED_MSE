import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.python.ops.gen_math_ops import square
from tqdm import tqdm

from load_dataset import load_test_data
from model import dped_g
import utils
import vgg
import os
import sys

import niqe
import lpips_tf

dataset_dir, test_dir, model_dir, result_dir,\
    dslr_dir, phone_dir, over_dir, under_dir, triple_exposure, up_exposure, down_exposure,\
    arch, level, norm_gen, num_maps_base, flat, orig_model, rand_param, restore_iter,\
    img_h, img_w, use_gpu, save_model, test_image = utils.process_test_model_args(sys.argv)

if flat:
    FAC_PATCH = 1
    PATCH_DEPTH = 1
else:
    FAC_PATCH = 2
    PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
elif up_exposure or down_exposure:
    PATCH_DEPTH *= 2

LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
PATCH_WIDTH = 256//FAC_PATCH
PATCH_HEIGHT = 256//FAC_PATCH
TARGET_WIDTH = int(PATCH_WIDTH * FAC_PATCH)
TARGET_HEIGHT = int(PATCH_HEIGHT * FAC_PATCH)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
if restore_iter == 0:
    restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
                for model_file in os.listdir(model_dir)
                if model_file.startswith("DPED_")])))
    restore_iters = reversed(restore_iters)
else:
    restore_iters = [restore_iter]
batch_size = 10
use_gpu = True

print("Loading testing data...")
test_data, test_answ = load_test_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat)
print("Testing data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

time_start = datetime.now()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None
with tf.compat.v1.Session(config=config) as sess:
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    enhanced = dped_g(phone_, flat=flat, norm=norm_gen)
    saver = tf.compat.v1.train.Saver()

    dslr_gray = tf.image.rgb_to_grayscale(dslr_)
    enhanced_gray = tf.image.rgb_to_grayscale(enhanced)

    ## PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list = [loss_psnr]
    loss_text = ["loss_psnr"]

    # MSE loss
    loss_mse = tf.reduce_mean(tf.math.squared_difference(enhanced, dslr_))
    loss_list.append(loss_mse)
    loss_text.append("loss_mse")

    ## L1 loss
    loss_l1 = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced, dslr_)))
    loss_list.append(loss_l1)
    loss_text.append("loss_l1")

    ## Color loss
    # enhanced_blur = utils.blur(enhanced)
    # dslr_blur = utils.blur(dslr_)
    # loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
    # loss_list.append(loss_color)
    # loss_text.append("loss_color")

    ## SSIM loss
    loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(enhanced_gray, dslr_gray, 1.0))
    loss_list.append(loss_ssim)
    loss_text.append("loss_ssim")

    # MS-SSIM loss
    loss_ms_ssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(enhanced_gray, dslr_gray, 1.0))
    loss_list.append(loss_ms_ssim)
    loss_text.append("loss_ms_ssim")

    ## Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_content = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_content)
    loss_text.append("loss_content")

    ## LPIPS
    loss_lpips = tf.reduce_mean(lpips_tf.lpips(enhanced, dslr_, net='vgg'))
    loss_list.append(loss_lpips)
    loss_text.append("loss_lpips")
    
    ## Huber loss
    delta = 1
    abs_error = tf.abs(tf.math.subtract(enhanced, dslr_))
    quadratic = tf.math.minimum(abs_error, delta)
    linear = tf.math.subtract(abs_error, quadratic)
    loss_huber = tf.reduce_mean(0.5*tf.math.square(quadratic)+linear)
    loss_list.append(loss_huber)
    loss_text.append("loss_huber")

    # ## Total variation loss
    # loss_tv = tf.reduce_mean(tf.image.total_variation(enhanced))
    # loss_list.append(loss_tv)
    # loss_text.append("loss_tv")


    niqe = niqe.create_evaluator()

    control_niqe = 0.0
    for i, restore_iter in enumerate(restore_iters):
        saver.restore(sess, model_dir + "DPED_iteration_" + str(restore_iter) + ".ckpt")
        test_losses_gen = np.zeros((1, len(loss_text)))
        metric_niqe = 0.0
        for j in tqdm(range(num_test_batches)):

            be = j * batch_size
            en = (j+1) * batch_size

            phone_images = test_data[be:en]
            dslr_images = test_answ[be:en]

            [losses, enhanced_images] = sess.run([loss_list, enhanced], feed_dict={phone_: phone_images, dslr_: dslr_images})
            test_losses_gen += np.asarray(losses) / num_test_batches

            # metric_niqe += niqe.evaluate(enhanced_images, enhanced_images) / num_test_batches
            # if i == 0:
            #     control_niqe += niqe.evaluate(dslr_images, dslr_images) / num_test_batches

        logs_gen = "Losses - iter: " + str(restore_iter) + "-> "
        for idx, loss in enumerate(loss_text):
            logs_gen += "%s: %.6g; " % (loss, test_losses_gen[0][idx])
        logs_gen += "niqe: %.6g; control-niqe: %.6g" % (metric_niqe, control_niqe)
        logs_gen += '\n'
        print(logs_gen)

        logs = open(model_dir + model_dir.split('/')[0] + ".txt", "a+")
        logs.write(logs_gen)
        logs.write('\n')
        logs.close()
print('total test time:', datetime.now() - time_start)