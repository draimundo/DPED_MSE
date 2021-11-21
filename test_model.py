###########################################
# Run the trained model on testing images #
###########################################

import numpy as np
import tensorflow as tf
import imageio
import sys
import os
import rawpy

from model import dped_g
import utils

from tqdm import tqdm
from datetime import datetime
from load_dataset import extract_bayer_channels

dataset_dir, test_dir, model_dir, result_dir, arch, LEVEL, inst_norm, num_maps_base,\
    orig_model, rand_param, restore_iter, IMAGE_HEIGHT, IMAGE_WIDTH, use_gpu, save_model, test_image = \
        utils.process_test_model_args(sys.argv)

DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
MAX_SCALE = float(1) / (2 ** (5 - 1))
IMAGE_HEIGHT, IMAGE_WIDTH = 1500, 2000

IMAGE_HCROP= int(np.floor(IMAGE_HEIGHT * MAX_SCALE)/MAX_SCALE)
IMAGE_WCROP = int(np.floor(IMAGE_WIDTH * MAX_SCALE)/MAX_SCALE)

TARGET_HEIGHT = int(np.floor(IMAGE_HCROP * DSLR_SCALE))
TARGET_WIDTH = int(np.floor(IMAGE_WCROP * DSLR_SCALE))

PATCH_HEIGHT = int(np.floor(IMAGE_HCROP*DSLR_SCALE)/DSLR_SCALE)
PATCH_WIDTH = int(np.floor(IMAGE_WCROP*DSLR_SCALE)/DSLR_SCALE)

TARGET_DEPTH = 3
PATCH_DEPTH = 4

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None

restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
            for model_file in os.listdir(model_dir)
            if model_file.startswith("DPED_")])))
restore_iters = reversed(restore_iters)


with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])


    # generate enhanced image
    enhanced = dped_g(x_)


    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/mediatek_raw_normal/'

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
        In = extract_bayer_channels(In)

        images[i,...] = In[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
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