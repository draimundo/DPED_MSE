import tensorflow as tf
import numpy as np
import sys
import time

from datetime import datetime
from load_dataset_keras import image_generator

import utils

from tqdm import tqdm

from losses_keras import *
from model_keras import *


dataset_dir = 'raw_images/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'
over_dir = 'mediatek_raw_over/'
under_dir = 'mediatek_raw_under/'
triple_exposure = True
LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
PATCH_WIDTH = 256
PATCH_HEIGHT = 256

train_size = 5


train_generator = image_generator(dataset_dir, dslr_dir, phone_dir, 'train/', PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir)
train_dataset = tf.data.Dataset.from_tensor_slices(train_generator.get_list())
train_dataset = train_dataset.shuffle(train_generator.length())
train_dataset = train_dataset.map(train_generator.read,
                                num_parallel_calls=-1)
train_dataset = train_dataset.map(train_generator.augment_image,
                                num_parallel_calls=-1)
train_dataset = train_dataset.batch(train_size)





val_generator = image_generator(dataset_dir, dslr_dir, phone_dir, 'val/', PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir)
val_dataset = tf.data.Dataset.from_tensor_slices(val_generator.get_list())
val_dataset = val_dataset.map(val_generator.read,
                                    num_parallel_calls=-1)

learning_rate = 5e-5
vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'


gen = dped_resnet(train_generator.size())
loss_fo = loss_fourier(PATCH_WIDTH, PATCH_HEIGHT)
gen.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    loss = loss_creator(vgg_dir, PATCH_WIDTH, PATCH_WIDTH, 200, 0, 0, 300, 2),
    metrics = [metr_psnr, loss_fo]
)

batch_size = 5
num_train_iters = 300000
eval_step = 100


hist = gen.fit(
    train_dataset,
    batch_size = batch_size,
    epochs = num_train_iters,
    validation_freq = eval_step,
    validation_data = val_dataset,
    workers=-1,
    use_multiprocessing=True
)