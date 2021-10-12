import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm

from load_dataset import load_test_data
from model import resnet
import utils
import vgg

LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
PATCH_WIDTH = 256//2
PATCH_HEIGHT = 256//2
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

dataset_dir = 'raw_images/'
model_dir = 'models/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'
vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
restore_iter = 84000
batch_size = 5
use_gpu = True

print("Loading testing data...")
test_data, test_answ = load_test_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
print("Testing data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

time_start = datetime.now()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None
with tf.compat.v1.Session(config=config) as sess:
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    name_model = "resnet"
    enhanced_ = resnet(phone_)
    saver = tf.compat.v1.train.Saver()
    name_model_full = name_model + "_iteration_" + str(restore_iter)
    saver.restore(sess, model_dir + name_model_full + ".ckpt")

    ## PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced_, dslr_, 1.0))
    loss_list = [loss_psnr]
    loss_text = ["loss_psnr"]

    ## Color loss
    enhanced_blur = utils.blur(enhanced_)
    dslr_blur = utils.blur(dslr_)
    loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
    loss_list.append(loss_color)
    loss_text.append("loss_color")

    ## SSIM loss
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced_, dslr_, 1.0))
    loss_list.append(loss_ssim)
    loss_text.append("loss_ssim")

    ## Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced_ * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_content = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_content)
    loss_text.append("loss_content")

    test_losses_gen = np.zeros((1, len(loss_text)))
    for j in tqdm(range(num_test_batches)):

        be = j * batch_size
        en = (j+1) * batch_size

        phone_images = test_data[be:en]
        dslr_images = test_answ[be:en]

        losses = sess.run(loss_list, feed_dict={phone_: phone_images, dslr_: dslr_images})
        test_losses_gen += np.asarray(losses) / num_test_batches

logs_gen = "Losses -> "
for idx, loss in enumerate(loss_text):
    logs_gen += "%s: %.4g; " % (loss, test_losses_gen[0][idx])
logs_gen += '\n'
print(logs_gen)
print('total test time:', datetime.now() - time_start)