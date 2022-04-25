# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from cgi import test
import numpy as np
import tensorflow as tf
import imageio
from tqdm import tqdm
import os
import imageio
import utils
import sys


from model import switch_model
from datetime import datetime
from load_dataset import extract_bayer_channels

dataset_dir, result_dir, vgg_dir, dslr_dir, phone_dir, model_dir, over_dir, under_dir,\
    triple_exposure, up_exposure, down_exposure, restore_iter, img_h, img_w,\
    activation, end_activation, norm_gen, flat, mix_input, onebyone, model_type, upscale,\
    num_feats, num_blocks, use_gpu = utils.process_test_model_args(sys.argv)

LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
MAX_SCALE = float(1) / (2 ** (5 - 1))
IMAGE_HEIGHT, IMAGE_WIDTH = 3000, 4000
TARGET_DEPTH = 3
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

TARGET_HEIGHT = IMAGE_HEIGHT
TARGET_WIDTH = IMAGE_WIDTH

PATCH_HEIGHT = int(np.floor(IMAGE_HEIGHT//FAC_PATCH))
PATCH_WIDTH = int(np.floor(IMAGE_WIDTH//FAC_PATCH))

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == False else None

if not os.path.isdir("results/full-resolution/"+ result_dir):
    os.makedirs("results/full-resolution/"+ result_dir, exist_ok=True)

if restore_iter == 0:
    restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
            for model_file in os.listdir(model_dir)
            if model_file.startswith("DPED_")])))
    restore_iters = reversed(restore_iters)
else:
    restore_iters = [restore_iter]

with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])


    # generate enhanced image
    # Get the processed enhanced image
    enhanced = switch_model(x_, model_type, activation, norm_gen, flat, mix_input, onebyone, upscale, end_activation, num_feats, num_blocks)


    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/' + phone_dir
    test_dir_over = 'validation_full_resolution_visual_data/' + over_dir
    test_dir_under = 'validation_full_resolution_visual_data/' + under_dir

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(imageio.imread((test_dir_full + photo)))
        if not flat:
            In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread((test_dir_over + photo)))
            Iu = np.asarray(imageio.imread((test_dir_full + photo)))
            if not flat:
                Io = extract_bayer_channels(Io)
                Iu = extract_bayer_channels(Iu)
            I = np.dstack((In, Io, Iu))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        elif up_exposure:
            Io = np.asarray(imageio.imread((test_dir_over + photo)))
            if not flat:
                Io = extract_bayer_channels(Io)

            I = np.dstack((In, Io))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        elif down_exposure:
            Iu = np.asarray(imageio.imread((test_dir_full + photo)))
            if not flat:
                Iu = extract_bayer_channels(Iu)

            I = np.dstack((In, Iu))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        else:
            I = In
            if flat:
                images[i,..., 0] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
            else:
                images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]

    print("Images loaded")
    # Run inference

    for restore_iter in tqdm(restore_iters):
        saver.restore(sess, model_dir + "DPED_iteration_" + str(restore_iter) + ".ckpt")
        
        for i, photo in enumerate(test_photos):
            enhanced_tensor = sess.run(enhanced, feed_dict={x_: [images[i,...]]})
            enhanced_image = np.reshape(enhanced_tensor, [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            imageio.imwrite("results/full-resolution/"+ result_dir + photo_name +
                        "_iteration_" + str(restore_iter) + ".png", enhanced_image)