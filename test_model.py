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

from tqdm import tqdm
from datetime import datetime
from load_dataset import extract_bayer_channels

dataset_dir, test_dir, model_dir, result_dir, arch, LEVEL, inst_norm, num_maps_base,\
    orig_model, rand_param, restore_iter, IMAGE_HEIGHT, IMAGE_WIDTH, use_gpu, save_model, test_image = \
        utils.process_test_model_args(sys.argv)
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None

restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
            for model_file in os.listdir(model_dir)
            if model_file.startswith("resnet_")])))


with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 4])

    # determine model name
    # generate enhanced image
    if arch == "resnet":
        name_model = "resnet"
        enhanced = resnet(x_)

    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/mediatek_raw_normal/'

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
        In = extract_bayer_channels(In)

        images[i,...] = In
    print("Images loaded")
    # Run inference

    for restore_iter in tqdm(restore_iters):
        saver.restore(sess, model_dir + "resnet_iteration_" + str(restore_iter) + ".ckpt")
        
        for i, photo in enumerate(test_photos):
            enhanced_tensor = sess.run(enhanced, feed_dict={x_: [images[i,...]]})
            enhanced_image = np.reshape(enhanced_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            imageio.imwrite("results/full-resolution/"+ result_dir + photo_name +
                        "_iteration_" + str(restore_iter) + ".png", enhanced_image)