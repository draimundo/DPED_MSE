##################################################
# Train a RAW-to-RGB model using training images #
##################################################

import tensorflow as tf
import imageio
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_val_data
from model import fourierDiscrim, resnet, adversarial
import utils
import vgg

from tqdm import tqdm
from skimage.filters import window


# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, \
save_mid_imgs, leaky, norm_gen, \
fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_color, fac_vgg, fac_texture, fac_fourier, fac_frequency \
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
tf.random.set_seed(0)

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    enhanced = resnet(phone_, leaky = leaky, instance_norm = norm_gen)

    # Losses
    dslr_gray = tf.image.rgb_to_grayscale(dslr_)
    enhanced_gray = tf.image.rgb_to_grayscale(enhanced)

    # MSE loss
    loss_mse = tf.reduce_mean(tf.math.squared_difference(enhanced, dslr_))
    loss_generator = loss_mse * fac_mse
    loss_list = [loss_mse]
    loss_text = ["loss_mse"]

    # L1 loss
    loss_l1 = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced, dslr_)))
    if fac_l1 > 0:
        loss_list.append(loss_l1)
        loss_text.append("loss_l1")
        loss_generator += loss_l1 * fac_l1

    # PSNR metric
    metric_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list.append(metric_psnr)
    loss_text.append("metric_psnr")

    # SSIM loss
    loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(enhanced_gray, dslr_gray, 1.0))
    if fac_ssim > 0:
        loss_generator += loss_ssim * fac_ssim
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")

    # MS-SSIM loss
    loss_ms_ssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(enhanced_gray, dslr_gray, 1.0))
    if fac_ms_ssim > 0:
        loss_generator += loss_ms_ssim * fac_ms_ssim
        loss_list.append(loss_ms_ssim)
        loss_text.append("loss_ms_ssim")

    # Color loss
    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_)
    loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
    if fac_color > 0:
        loss_generator += loss_color * fac_color
        loss_list.append(loss_color)
        loss_text.append("loss_color")

    # Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_vgg = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_vgg)
    loss_text.append("loss_vgg")
    if fac_vgg > 0:
        loss_generator += loss_vgg * fac_vgg

    ## Adversarial loss - discrim
    if fac_texture > 0:
        enhanced_gray_res = tf.reshape(enhanced_gray, [-1, TARGET_WIDTH * TARGET_HEIGHT])
        dslr_gray_res = tf.reshape(dslr_gray,[-1, TARGET_WIDTH * TARGET_HEIGHT])
        adv_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
        adversarial_ = tf.multiply(enhanced_gray_res, 1 - adv_) + tf.multiply(dslr_gray_res, adv_)
        adversarial_image = tf.reshape(adversarial_, [-1, TARGET_HEIGHT, TARGET_WIDTH, 1])
        discrim_predictions = adversarial(adversarial_image)
        discrim_target = tf.concat([adv_, 1 - adv_], 1)
        loss_discrim = -tf.reduce_mean(discrim_target * tf.math.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
        loss_texture = -loss_discrim
        correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
        discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        loss_generator += loss_texture * fac_texture
        loss_list.append(loss_texture)
        loss_text.append("loss_texture")

    ## Fourier losses (normal and adversarial)
    h2d = np.float32(window('hann', (TARGET_WIDTH, TARGET_HEIGHT)))
    hann2d = tf.stack([h2d,h2d,h2d],axis=2) #stack for 3 color channels
    enhanced_filter = tf.cast(tf.multiply(enhanced, hann2d),tf.complex64)
    dslr_filter = tf.cast(tf.multiply(enhanced, hann2d),tf.complex64)
    enhanced_fft = tf.signal.fft2d(tf.transpose(enhanced_filter, [0, 3, 1, 2]))
    dslr_fft = tf.signal.fft2d(tf.transpose(dslr_filter, [0, 3, 1, 2]))
    loss_fourier = 1/2 * tf.reduce_mean(tf.abs(tf.abs(enhanced_fft)-tf.abs(dslr_fft))) + \
                1/2 * tf.reduce_mean(tf.abs(tf.math.angle(enhanced_fft)-tf.math.angle(dslr_fft)))
    if fac_fourier > 0:
        loss_generator += loss_fourier * fac_fourier
        loss_list.append(loss_fourier)
        loss_text.append("loss_fourier")

    if fac_frequency > 0:
        enhanced_fft = tf.transpose(enhanced_fft,[0,2,3,1])
        dslr_fft = tf.transpose(dslr_fft,[0,2,3,1])
        adv_fourier = tf.compat.v1.placeholder(tf.complex64, [None, 1])
        adversarial_fourier = tf.multiply(enhanced_fft, 1 - adv_fourier) + tf.multiply(dslr_fft, adv_fourier)
        adversarial_fourier_image = tf.reshape(adversarial_fourier, [-1, TARGET_HEIGHT, TARGET_WIDTH, -1])
        discrim_fourier_predictions = fourierDiscrim(adversarial_fourier)
        discrim_fourier_target = tf.concat([adv_fourier, 1 - adv_fourier], 1)
        loss_discrim_fourier = -tf.reduce_mean(discrim_fourier_target * tf.math.log(tf.clip_by_value(discrim_fourier_predictions, 1e-10, 1.0)))
        loss_frequency = -loss_discrim_fourier
        correct_predictions_fourier = tf.equal(tf.argmax(discrim_fourier_predictions, 1), tf.argmax(discrim_fourier_target, 1))
        discim_accuracy_fourier = tf.reduce_mean(tf.cast(correct_predictions_fourier, tf.float32))
        loss_generator += loss_frequency * fac_frequency
        loss_list.append(loss_frequency)
        loss_text.append("loss_frequency")

    ## Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")

    # Optimize network parameters
    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    if fac_texture > 0:
        discriminator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("discriminator")]
        train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)
        all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

    if fac_frequency > 0:
        fourier_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("fourierDiscrim")]
        train_step_fourier = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim_fourier, var_list=fourier_vars)
        all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    if restore_iter > 0: # restore the variables/weights
        name_model_restore_full = "DPED" + "_iteration_" + str(restore_iter)
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
    if fac_texture > 0:
        train_acc_discrim = 0.0
    if fac_frequency > 0:
        train_acc_discrim_fourier = 0.0
    
    for i in tqdm(range(iter_start, num_train_iters + 1), miniters=100):
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

        traindict = {phone_: phone_images, dslr_: dslr_images}
        if fac_texture > 0:
            traindict[adv_] = all_zeros
        if fac_frequency > 0:
            traindict[adv_fourier] = all_zeros


        # Training step
        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict=traindict)
        training_loss += loss_temp / eval_step

        # Train discrimintator
        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]
        if fac_texture > 0:
            traindict[adv_] = swaps
            [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                            feed_dict=traindict)
            train_acc_discrim += accuracy_temp / eval_step
        if fac_frequency > 0:
            traindict[adv_fourier] = swaps
            [accuracy_temp, temp] = sess.run([discim_accuracy_fourier, train_step_fourier],
                                            feed_dict=traindict)
            train_acc_discrim += accuracy_temp / eval_step

        #  Evaluate model
        if i % eval_step == 0:
            val_losses_gen = np.zeros((1, len(loss_text)))

            val_acc_disc = 0.0
            val_acc_fourier = 0.0

            for j in range(num_val_batches):
                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                valdict = {phone_: phone_images, dslr_: dslr_images}
                toRun = [loss_list]
                if fac_texture > 0:
                    valdict[adv_] = swaps
                if fac_frequency > 0:
                    valdict[adv_fourier] = swaps

                losses = sess.run(toRun, feed_dict=valdict)
                val_losses_gen += np.asarray(losses) / num_val_batches
                if fac_texture > 0:
                    accuracy_disc = sess.run(discim_accuracy, feed_dict=valdict)
                    val_acc_disc += accuracy_disc / num_val_batches
                if fac_frequency > 0:
                    accuracy_fourier = sess.run(discim_accuracy_fourier, feed_dict=valdict)
                    val_acc_fourier += accuracy_fourier / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, training_loss)
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses_gen[0][idx])
            if fac_texture > 0:
                logs_gen += "\n | discriminator accuracy: %.4g" % (val_acc_disc)
            if fac_frequency > 0:  
                logs_gen += "\n |Fourier discriminator: %.4g" % (val_acc_fourier)

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
                        imageio.imwrite(result_dir + "DPED_img_" + str(idx) + ".jpg",
                                        before_after)
                    idx += 1

            # Saving the model that corresponds to the current iteration
            saver.save(sess, model_dir + "DPED_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

            training_loss = 0.0


        # Loading new training data
        if i % 1000 == 0 and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

    print('total train/eval time:', datetime.now() - time_start)