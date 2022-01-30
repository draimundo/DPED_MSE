##################################################
# Train a RAW-to-RGB model using training images #
##################################################

import tensorflow as tf
import imageio
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_val_data
from model import dped_g, resnext_g, fourier_d, texture_d, unet_d
import utils
import vgg
import lpips_tf

from tqdm import tqdm
from skimage.filters import window

from AdaBound import AdaBoundOptimizer
from AMSGrad import AMSGrad
from RAdam import RAdamOptimizer


# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
triple_exposure, up_exposure, down_exposure, over_dir, under_dir,\
patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, \
save_mid_imgs, leaky, norm_gen, norm_disc, flat, percentage, entropy, mix, optimizer,\
mix_input, onebyone, model_type, upscale,\
fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_color, fac_vgg, fac_texture, fac_fourier, fac_frequency, fac_lpips, fac_huber, fac_unet \
    = utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches
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


PATCH_WIDTH = patch_w//FAC_PATCH
PATCH_HEIGHT = patch_h//FAC_PATCH
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

LEVEL = 0
DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
TARGET_WIDTH = int(PATCH_WIDTH * FAC_PATCH)
TARGET_HEIGHT = int(PATCH_HEIGHT * FAC_PATCH)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)
tf.random.set_seed(0)

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    if model_type == 'resnext':
        enhanced = resnext_g(phone_, leaky = leaky, norm = norm_gen, flat = flat, mix_input = mix_input, onebyone = onebyone, upscale = upscale)
    elif model_type == 'dped':
        enhanced = dped_g(phone_, leaky = leaky, norm = norm_gen, flat = flat, mix_input = mix_input, onebyone = onebyone, upscale = upscale)
    else:
        raise NotImplementedError("Missing model " + model_type)

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

    # Huber loss
    delta = 1
    abs_error = tf.abs(tf.math.subtract(enhanced, dslr_))
    quadratic = tf.math.minimum(abs_error, delta)
    linear = tf.math.subtract(abs_error, quadratic)
    loss_huber = tf.reduce_mean(0.5*tf.math.square(quadratic)+linear)
    if fac_huber > 0:
        loss_generator += loss_huber * fac_huber
        loss_list.append(loss_huber)
        loss_text.append("loss_huber")

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
        adv_real = dslr_gray
        adv_fake = enhanced_gray

        pred_real = texture_d(adv_real, activation=False)
        pred_fake = texture_d(adv_fake, activation=False)

        loss_texture_g = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_fake - pred_real), 1e-10, 1.0)))
        loss_texture_d = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_real - pred_fake), 1e-10, 1.0)))

        loss_generator += loss_texture_g * fac_texture
        loss_list.append(loss_texture_g)
        loss_text.append("loss_texture")

    ## Fourier losses (normal and adversarial)
    h2d = np.float32(window('hann', (TARGET_WIDTH, TARGET_HEIGHT)))
    hann2d = tf.stack([h2d,h2d,h2d],axis=2) #stack for 3 color channels

    enhanced_filter = tf.cast(tf.multiply(enhanced, hann2d),tf.float32)
    dslr_filter = tf.cast(tf.multiply(dslr_, hann2d),tf.float32)

    # from NHWC to NCHW and back, rfft2d performed on 2 innermost dimensions
    enhanced_fft = tf.signal.rfft2d(tf.transpose(enhanced_filter, [0, 3, 1, 2]))
    enhanced_fft = tf.transpose(enhanced_fft,[0,2,3,1])
    enhanced_abs = tf.abs(enhanced_fft)
    enhanced_angle = tf.math.angle(enhanced_fft)

    dslr_fft = tf.signal.rfft2d(tf.transpose(dslr_filter, [0, 3, 1, 2]))
    dslr_fft = tf.transpose(dslr_fft,[0,2,3,1])
    dslr_abs = tf.abs(dslr_fft)
    dslr_angle = tf.math.angle(dslr_fft)

    # diff_angle = tf.abs(tf.math.subtract(enhanced_angle, dslr_angle))
    # abs_angle = tf.minimum(
    #     diff_angle,
    #     tf.math.subtract(2*np.pi, diff_angle)
    # )
    
    loss_fourier = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced_fft, dslr_fft))) # + \
                #    1/2 * tf.reduce_mean(abs_angle)
    
    loss_list.append(loss_fourier)
    loss_text.append("loss_fourier")
    if fac_fourier > 0:
        loss_generator += loss_fourier * fac_fourier

    if fac_frequency > 0:
        frequency_real = tf.stack([dslr_abs, dslr_angle], axis=2)
        frequency_fake = tf.stack([enhanced_abs, enhanced_angle], axis=2)
        
        frequency_pred_real = fourier_d(frequency_real, activation=False)
        frequency_pred_fake = fourier_d(frequency_fake, activation=False)

        loss_frequency_g = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(frequency_pred_fake - frequency_pred_real), 1e-10, 1.0)))
        loss_frequency_d = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(frequency_pred_real - frequency_pred_fake), 1e-10, 1.0)))

        loss_generator += loss_frequency_g * fac_frequency
        loss_list.append(loss_frequency_g)
        loss_text.append("loss_frequency")

    if fac_unet > 0:
        adv_real = dslr_
        adv_fake = enhanced

        pred_real = unet_d(adv_real, activation=False, norm=norm_disc)
        pred_fake = unet_d(adv_fake, activation=False, norm=norm_disc)

        loss_unet_g = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_fake - pred_real), 1e-10, 1.0)))
        loss_unet_d = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_real - pred_fake), 1e-10, 1.0)))

        loss_generator += loss_unet_g * fac_unet
        loss_list.append(loss_unet_g)
        loss_text.append("loss_unet")


    ## LPIPS
    loss_lpips = tf.reduce_mean(lpips_tf.lpips(enhanced, dslr_, net='alex'))
    loss_list.append(loss_lpips)
    loss_text.append("loss_lpips")
    if fac_lpips > 0:
        loss_generator += loss_lpips * fac_lpips


    ## Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")


    # Optimize network parameters
    vars_dped_g = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    # if optimizer == "adamp":
    #     train_step_dped_g = AdamP(learning_rate=learning_rate).minimize(loss_no_args, var_list=vars_dped_g)
    # elif optimizer == "sgdp":
    #     train_step_dped_g = SGDP(learning_rate=learning_rate).minimize(loss_no_args, var_list=vars_dped_g)
    if optimizer == "adabound":
        train_step_dped_g = AdaBoundOptimizer(learning_rate=learning_rate).minimize(loss_generator, var_list=vars_dped_g)
    elif optimizer == "amsbound":
        train_step_dped_g = AdaBoundOptimizer(learning_rate=learning_rate, amsbound=True).minimize(loss_generator, var_list=vars_dped_g)
    elif optimizer == "amsgrad":
        train_step_dped_g = AMSGrad(learning_rate=learning_rate).minimize(loss_generator, var_list=vars_dped_g)
    elif optimizer == "radam":
        train_step_dped_g = RAdamOptimizer(learning_rate=learning_rate).minimize(loss_generator, var_list=vars_dped_g)
    elif optimizer == "adam":
        train_step_dped_g = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=vars_dped_g)
    else:
        print("Optimizer not found -> using Adam")
        train_step_dped_g = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=vars_dped_g)

    if fac_texture > 0:
        loss_texture_g_ = 0.0
        n_texture_d_ = 0.0
        lr_texture_d = learning_rate
        vars_texture_d = [v for v in tf.compat.v1.global_variables() if v.name.startswith("texture_d")]
        train_step_texture_d = RAdamOptimizer(lr_texture_d/10000.0).minimize(loss_texture_d, var_list=vars_texture_d)

    if fac_frequency > 0:
        loss_frequency_g_ = 0.0
        n_frequency_d_ = 0.0
        lr_frequency_d = learning_rate
        vars_frequency_d = [v for v in tf.compat.v1.global_variables() if v.name.startswith("fourier_d")]
        train_step_frequency_d = RAdamOptimizer(lr_frequency_d/10000.0).minimize(loss_frequency_d, var_list=vars_frequency_d)

    if fac_unet > 0:
        loss_unet_g_ = 0.0
        n_unet_d_ = 0.0
        lr_unet_d = learning_rate
        vars_unet_d = [v for v in tf.compat.v1.global_variables() if v.name.startswith("unet_d")]
        train_step_unet_d = RAdamOptimizer(lr_unet_d/10000.0).minimize(loss_unet_d, var_list=vars_unet_d)

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=vars_dped_g, max_to_keep=1000)

    if restore_iter > 0: # restore the variables/weights
        name_model_restore_full = "DPED" + "_iteration_" + str(restore_iter)
        print("Restoring Variables from:", name_model_restore_full)
        saver.restore(sess, model_dir + name_model_restore_full + ".ckpt")

    # Loading training and validation data
    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat, percentage, entropy, mix)
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

    loss_dped_g_ = 0.0
    
    for i in tqdm(range(iter_start, num_train_iters + 1), miniters=100):
        # Train texture discriminator
        if fac_texture > 0:
            idx_texture_d = np.random.randint(0, train_size, batch_size)
            phone_texture_d = train_data[idx_texture_d]
            dslr_texture_d = train_answ[idx_texture_d]

            feed_texture_d = {phone_: phone_texture_d, dslr_: dslr_texture_d}
            [loss_g, loss_d] = sess.run([loss_texture_g, loss_texture_d], feed_dict=feed_texture_d)

            if loss_g < 3*loss_d:
                [loss_temp, temp] = sess.run([loss_texture_d, train_step_texture_d], feed_dict=feed_texture_d)
                n_texture_d_ += 1

        # Train frequency discriminator
        if fac_frequency > 0:
            idx_frequency_d = np.random.randint(0, train_size, batch_size)
            phone_frequency_d = train_data[idx_frequency_d]
            dslr_frequency_d = train_answ[idx_frequency_d]

            feed_frequency_d = {phone_: phone_frequency_d, dslr_: dslr_frequency_d}
            [loss_g, loss_d] = sess.run([loss_frequency_g, loss_frequency_d], feed_dict=feed_frequency_d)

            if loss_g < 3*loss_d:
                [loss_temp, temp] = sess.run([loss_frequency_d, train_step_frequency_d], feed_dict=feed_frequency_d)
                n_frequency_d_ += 1

        # Train unet discriminator
        if fac_unet > 0:
            idx_unet_d = np.random.randint(0, train_size, batch_size)
            phone_unet_d = train_data[idx_unet_d]
            dslr_unet_d = train_answ[idx_unet_d]

            feed_unet_d = {phone_: phone_unet_d, dslr_: dslr_unet_d}
            [loss_g, loss_d] = sess.run([loss_unet_g, loss_unet_d], feed_dict=feed_unet_d)

            if loss_g < 3*loss_d:
                [loss_temp, temp] = sess.run([loss_unet_d, train_step_unet_d], feed_dict=feed_unet_d)
                n_unet_d_ += 1


        # Train generator
        idx_g = np.random.randint(0, train_size, batch_size)
        phone_g = train_data[idx_g]
        dslr_g = train_answ[idx_g]

        for k in range(batch_size):
            random_rotate = np.random.randint(1, 100) % 4
            phone_g[k] = np.rot90(phone_g[k], random_rotate)
            dslr_g[k] = np.rot90(dslr_g[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2
            if random_flip == 1:
                phone_g[k] = np.flipud(phone_g[k])
                dslr_g[k] = np.flipud(dslr_g[k])

        feed_g = {phone_: phone_g, dslr_: dslr_g}
        [loss_temp, temp] = sess.run([loss_generator, train_step_dped_g], feed_dict=feed_g)
        loss_dped_g_ += loss_temp / eval_step

        #  Evaluate model
        if i % eval_step == 0:
            val_losses_g = np.zeros((1, len(loss_text)))
            if fac_texture > 0:
                val_loss_texture_d = 0.0
            if fac_frequency > 0:
                val_loss_frequency_d = 0.0
            if fac_unet > 0:
                val_loss_unet_d = 0.0

            for j in range(num_val_batches):
                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                valdict = {phone_: phone_images, dslr_: dslr_images}
                toRun = [loss_list]

                loss_temp = sess.run(toRun, feed_dict=valdict)
                val_losses_g += np.asarray(loss_temp) / num_val_batches

                if fac_texture > 0:
                    loss_temp = sess.run(loss_texture_d, feed_dict=valdict)
                    val_loss_texture_d += loss_temp / num_val_batches
                if fac_frequency > 0:
                    loss_temp = sess.run(loss_frequency_d, feed_dict=valdict)
                    val_loss_frequency_d += loss_temp / num_val_batches
                if fac_unet > 0:
                    loss_temp = sess.run(loss_unet_d, feed_dict=valdict)
                    val_loss_unet_d += loss_temp / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, loss_dped_g_)
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses_g[0][idx])
            if fac_texture > 0:
                logs_gen += " | texture_d loss: %.4g; n_texture_d: %.4g" % (val_loss_texture_d, n_texture_d_)
            if fac_frequency > 0:
                logs_gen += " | frequency_d loss: %.4g; n_frequency_d: %.4g" % (val_loss_frequency_d, n_frequency_d_)
            if fac_unet > 0:
                logs_gen += " | unet_d loss: %.4g; n_unet_d: %.4g" % (val_loss_unet_d, n_unet_d_)

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

            loss_dped_g_ = 0.0
            if fac_texture > 0:
                n_texture_d_ = 0.0
            if fac_frequency > 0:
                n_frequency_d_ = 0.0
            if fac_unet > 0:
                n_unet_d_ = 0.0

        # Loading new training data
        if i % 1000 == 0  and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat, percentage, entropy, mix)
    print('total train/eval time:', datetime.now() - time_start)