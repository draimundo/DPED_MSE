##################################################
# Train a RAW-to-RGB model using training images #
##################################################

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import imageio
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_val_data
from model import resnet, adversarial
import utils
import vgg

from tqdm import tqdm


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

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # determine model name
    # Get the processed enhanced image

    if arch == "resnet":
        name_model = "resnet"
        enhanced = resnet(phone_, leaky = leaky, instance_norm = norm_gen)


    # Losses
    # MSE loss
    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])
    loss_mse = 2*tf.nn.l2_loss(dslr_flat - enhanced_flat)/(TARGET_SIZE * batch_size)
    loss_generator = loss_mse * fac_mse
    loss_list = [loss_mse]
    loss_text = ["loss_mse"]

    # texture (adversarial) loss
    # if fac_texture > 0:
    #     enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, TARGET_WIDTH * TARGET_HEIGHT])
    #     dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_),[-1, TARGET_WIDTH * TARGET_HEIGHT])
    #     # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    #     adv_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
    #     adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    #     adversarial_image = tf.reshape(adversarial_, [-1, TARGET_HEIGHT, TARGET_WIDTH, 1])
    #     discrim_predictions = adversarial(adversarial_image)
    #     discrim_target = tf.concat([adv_, 1 - adv_], 1)
    #     loss_discrim = -tf.reduce_sum(discrim_target * tf.compat.v1.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    #     loss_texture = -loss_discrim
    #     correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    #     discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    #     loss_generator = loss_generator + loss_texture * fac_texture
    #     loss_list.append(loss_texture)
    #     loss_text.append("loss_texture")
    
    # color loss
    if fac_color > 0:
        enhanced_blur = utils.blur(enhanced)
        dslr_blur = utils.blur(dslr_)
        loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)
        loss_generator += loss_color * fac_color
        loss_list.append(loss_color)
        loss_text.append("loss_color")

    # PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list.append(loss_psnr)
    loss_text.append("loss_psnr")

    # SSIM loss
    if fac_ssim > 0:
        loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))
        loss_generator += (1 - loss_ssim) * fac_ssim
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")

    # MS-SSIM loss
    #loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    # Content loss
    if fac_content > 0:
        CONTENT_LAYER = 'relu5_4'

        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))

        content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
        loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
        loss_generator += loss_content * fac_content
        loss_list.append(loss_content)
        loss_text.append("loss_content")

    # Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")

    # Optimize network parameters
    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    # if fac_texture > 0:
    #     discriminator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("discriminator")]
    #     train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    if restore_iter > 0: # restore the variables/weights
        name_model_restore = name_model

        name_model_restore_full = name_model_restore + "_iteration_" + str(restore_iter)
        print("Restoring Variables from:", name_model_restore_full)
        saver.restore(sess, model_dir + name_model_restore_full + ".ckpt")

    # Loading training and validation data
    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    if save_mid_imgs:
        visual_crops_ids = np.random.randint(0, VAL_SIZE, batch_size)
        visual_val_crops = val_data[visual_crops_ids, :]
        visual_target_crops = val_answ[visual_crops_ids, :]


    print("Training network...")

    iter_start = restore_iter+1 if restore_iter > 0 else 0
    logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "w+")
    logs.close()

    training_loss = 0.0
    train_acc_discrim = 0.0

    name_model_save = name_model
    
    for i in tqdm(range(iter_start, num_train_iters + 1), miniters=100):
        name_model_save_full = name_model_save + "_iteration_" + str(i)

        # Train genarator
        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        # Data augmentation: random flips and rotations
        for k in range(batch_size):

            random_rotate = np.random.randint(1, 100) % 4
            phone_images[k] = np.rot90(phone_images[k], random_rotate)
            dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2

            if random_flip == 1:
                phone_images[k] = np.flipud(phone_images[k])
                dslr_images[k] = np.flipud(dslr_images[k])

        # Training step
        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: dslr_images})
        training_loss += loss_temp / eval_step

        # Train discrimintator
    
        # idx_train = np.random.randint(0, train_size, batch_size)

        # # generate image swaps (dslr or enhanced) for discriminator
        # swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        # phone_images = train_data[idx_train]
        # dslr_images = train_answ[idx_train]

        # [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
        #                                 feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
        # train_acc_discrim += accuracy_temp / eval_step

        if i % eval_step == 0:

            # Evaluate model
            val_losses = np.zeros((1, len(loss_text)))

            for j in range(num_val_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                [enhanced_crops, losses] = sess.run([enhanced, loss_list], \
                                feed_dict={phone_: phone_images, dslr_: dslr_images})

                val_losses += np.asarray(losses) / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, training_loss, val_losses[0][0])
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses[0][idx])
            
            logs_gen += "\n"

            print(logs_gen)

            # Save the results to log file
            logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # Optional: save visual results for several validation image crops
            if save_mid_imgs:
                enhanced_crops = sess.run(enhanced, feed_dict={phone_: visual_val_crops, dslr_: dslr_images})

                idx = 0
                for crop in enhanced_crops:
                    if idx < 4:
                        before_after = np.hstack((crop,
                                        np.reshape(visual_target_crops[idx], [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])))
                        imageio.imwrite(result_dir + name_model_save_full + "_img_" + str(idx) + ".jpg",
                                        before_after)
                    idx += 1

            # Saving the model that corresponds to the current iteration
            saver.save(sess, model_dir + name_model_save_full + ".ckpt", write_meta_graph=False)

            training_loss = 0.0

        #if i % test_step == 0 and i > 0:


        # Loading new training data
        if i % 1000 == 0 and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

    print('total train/eval time:', datetime.now() - time_start)