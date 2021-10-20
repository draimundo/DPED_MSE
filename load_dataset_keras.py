
import tensorflow as tf
from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os


def show(input):
  plt.figure()
  plt.imshow(input[1][0])
  plt.axis('off')
  plt.pause(500)

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = tf.cast(tf.concat([ch_B, ch_Gb, ch_R, ch_Gr], axis=2), dtype = tf.float32)
    RAW_norm = RAW_combined / (4 * 255)

    return RAW_norm

class image_generator():
  def __init__(self, dataset_dir, dslr_dir, phone_dir, type_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure = False, over_dir = None, under_dir = None):
    self.directory_dslr = dataset_dir + type_dir + dslr_dir
    self.directory_phone = dataset_dir + type_dir + phone_dir
    self.triple_exposure = triple_exposure
    if triple_exposure:
      self.directory_over = dataset_dir + type_dir + over_dir
      self.directory_under = dataset_dir + type_dir + under_dir

    self.PATCH_WIDTH = PATCH_WIDTH
    self.PATCH_HEIGHT = PATCH_HEIGHT
    self.DSLR_SCALE = DSLR_SCALE

    self.dslr_list = [name for name in os.listdir(self.directory_dslr)
                               if os.path.isfile(os.path.join(self.directory_dslr, name))]
  
  def get_list(self):
    return self.dslr_list

  def length(self):
    return len(self.dslr_list)

  def size(self):
    img = self.dslr_list[0]
    return self.read(img)[0].shape

  def call(self):
    for img in self.dslr_list:
      yield img

  def augment_image(self, data, answ):
    random_rotate = np.random.randint(1, 100) % 4
    random_flip = np.random.randint(1, 100) % 2

    data = tf.image.rot90(data, random_rotate)
    answ = tf.image.rot90(answ, random_rotate)

    if random_flip == 1:
      data = tf.image.flip_up_down(data)
      answ = tf.image.flip_up_down(answ)
    return data, answ

  def read(self, img):
    In = tf.image.decode_png(tf.io.read_file(self.directory_phone + img))
    data = extract_bayer_channels(In)

    if self.triple_exposure:
      Io = tf.image.decode_png(tf.io.read_file(self.directory_over + img))
      Io = extract_bayer_channels(Io)
      Iu = tf.image.decode_png(tf.io.read_file(self.directory_under + img))
      Iu = extract_bayer_channels(Iu)
      data = tf.cast(tf.concat([data, Io, Iu], axis=2), dtype = tf.float32)

    answ = tf.cast(tf.image.decode_png(tf.io.read_file(self.directory_dslr + img)), dtype = tf.float32) / 255
    return data, answ


if __name__ == "__main__":
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

  train_generator = image_generator(dataset_dir, dslr_dir, phone_dir, 'train/', PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir)
  train_dataset = tf.data.Dataset.from_tensor_slices(train_generator.get_list())
  train_dataset = train_dataset.shuffle(train_generator.length())
  train_dataset = train_dataset.batch(15)
  # train_dataset = train_dataset.map(train_generator.read,
  #                                   num_parallel_calls=-1)
  # train_dataset = train_dataset.map(train_generator.augment_image,
  #                                   num_parallel_calls=-1)
  for i in range(6):
    print(next(iter(train_dataset)))
  

# train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed = 0)
# train_dataset = train_dataset.batch(BATCH_SIZE)