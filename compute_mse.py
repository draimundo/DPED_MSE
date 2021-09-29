###########################################
# Run the trained model on testing images #
###########################################

import numpy as np
import tensorflow as tf
import imageio
import sys
import os
import rawpy

from model import resnet
import utils

from datetime import datetime
from load_dataset import extract_bayer_channels

def loss_psnr(loss_mse):
    return 20 * utils.log10(1.0 / tf.sqrt(loss_mse))


dataset_dir, test_dir, model_dir, result_dir, arch, LEVEL, inst_norm, num_maps_base,\
    orig_model, rand_param, restore_iter, IMAGE_HEIGHT, IMAGE_WIDTH, use_gpu, save_model, test_image = \
        utils.process_test_model_args(sys.argv)
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))

# Disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None

with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # determine model name
    if arch == "punet":
        name_model = "punet"

    # Placeholders for test data
    raw_img = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 4])

    # generate enhanced image
    # if arch == "punet":
    #     enhanced = PUNET(x_, instance_norm=inst_norm, instance_norm_level_1=False, num_maps_base=num_maps_base)
    enhanced = resnet(raw_img)

    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    if orig_model: # load official pre-trained weights
        model_dir = "models/original/"
        name_model_full = name_model + '_pretrained'
        saver.restore(sess, model_dir + name_model_full + ".ckpt")
    else:
        if rand_param: # use random weights
            name_model_full = name_model
            global_vars = [v for v in tf.compat.v1.global_variables()]
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(var_list=global_vars)
        else: # load previous/restored pre-trained weights
            name_model_full = name_model + "_iteration_" + str(restore_iter)
            saver.restore(sess, model_dir + name_model_full + ".ckpt")

    # Processing test images
    if test_image:
        test_dir = '/validation_full_resolution_visual_data/'

        raw_dir = test_dir + 'mediatek_raw_normal/'
        raw_photos = [f for f in os.listdir(raw_dir) if os.path.isfile(raw_dir + f)]
        raw_photos.sort()


        isp_img = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        isp_dir = test_dir + 'mediatek_isp/'
        isp_photos = [f for f in os.listdir(isp_dir) if os.path.isfile(isp_dir + f)]
        isp_photos.sort()


        dslr_img = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        dslr_dir = test_dir +'fuji/'
        dslr_photos = [f for f in os.listdir(dslr_dir) if os.path.isfile(dslr_dir + f)]
        dslr_photos.sort()

        losses_psnr_enhanced = np.zeros(len(raw_photos))
        losses_psnr_isp = np.zeros(len(isp_photos))

        for idx, photo in enumerate(dslr_photos):

            print("Processing image " + photo)

            I = np.asarray(rawpy.imread((dslr_dir + dslr_photos[idx])))
            I = extract_bayer_channels(I)

            I = I[0:IMAGE_HEIGHT//2, 0:IMAGE_WIDTH//2, :]
            I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

            # Run inference
            enhanced_tensor = sess.run(enhanced, feed_dict={raw_img: I})
            enhanced_image = np.reshape(enhanced_tensor, [int(I.shape[1] * DSLR_SCALE), int(I.shape[2] * DSLR_SCALE), 3])

            dslr_image = np.asarray(imageio.imread(dslr_dir + dslr_photos[idx])) / 255.0
            isp_image = np.asarray(imageio.imread(isp_dir + isp_photos[idx])) / 255.0

            losses_psnr_enhanced[idx] = loss_psnr(np.square(np.subtract(enhanced_image,dslr_image)).mean())
            losses_psnr_isp[idx] = loss_psnr(np.square(np.subtract(isp_image,dslr_image)).mean())

            print("Photo " + photo + " enhanced PSNR: " + str(losses_psnr_enhanced[idx]) + " isp PSNR: " + str(losses_psnr_isp[idx]))

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            imageio.imwrite(result_dir + photo_name + "-" + name_model_full + ".png", enhanced_image)            

    print('total test time:', datetime.now() - time_start)
    print("Avg: enhanced MSE: " + str(np.mean(losses_psnr_enhanced)) + " isp MSE: " + str(np.mean(losses_psnr_isp)))
    # save model again (optional, but useful for MAI challenge)
    if save_model:

        saver.save(sess, model_dir + name_model_full + ".ckpt") # pre-trained weight + meta graph
        utils.export_pb(sess, 'output_l0', model_dir, name_model_full + ".pb") # frozen graph (i.e. protobuf)
        tf.compat.v1.summary.FileWriter(model_dir + name_model_full, sess.graph) # tensorboard
        print('model saved!')


