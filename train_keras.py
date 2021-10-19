import tensorflow as tf
import imageio
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_train_patch_exp, load_val_data

import utils

from tqdm import tqdm

from losses_keras import *
from model_keras import *


# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir,\
    arch, LEVEL, inst_norm, num_maps_base, restore_iter, patch_w, patch_h,\
        batch_size, train_size, learning_rate, eval_step, num_train_iters, save_mid_imgs, \
        leaky, norm_gen, fac_content, fac_mse, fac_ssim, fac_color, fac_texture \
        = utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches
PATCH_WIDTH = patch_w//2
PATCH_HEIGHT = patch_h//2
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)

time_start = datetime.now()

# Loading training and validation data
print("Loading training data...")
if exp_augmentation:
    train_data, train_answ = load_train_patch_exp(dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):
else:
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
print("Training data was loaded\n")

print("Loading validation data...")
val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
print("Validation data was loaded\n")

VAL_SIZE = val_data.shape[0]
num_val_batches = int(val_data.shape[0] / batch_size)

gen = dped_resnet(train_data[0].shape)
gen.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    loss = loss_creator(vgg_dir, patch_w, patch_h, 200, 0, 0.5, 300, 2),
    metrics = 
)
