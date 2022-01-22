#####################
# Utility functions #
#####################

from functools import reduce
import tensorflow as tf
import scipy.stats as st
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [100000, 35000, 20000, 20000, 5000, 5000]


def process_command_args(arguments):

    # Specifying the default parameters for training/validation

    # --- data path ---
    dataset_dir = 'raw_images/'
    result_dir = 'results/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    model_dir = 'test/'
    # --- model weights ---
    restore_iter = 0
    # --- input size ---
    patch_w = 256 # default size for MAI dataset
    patch_h = 256 # default size for MAI dataset
    # --- training options ---
    batch_size = 30
    train_size = 5000
    learning_rate = 5e-5
    eval_step = 1000
    num_train_iters = 100000
    # --- more options ---
    save_mid_imgs = False
    leaky = True
    norm_gen = 'instance'
    norm_disc = 'instance'
    flat = 0
    percentage = 100
    entropy='no'
    mix = 0
    optimizer='radam'

    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    triple_exposure = False
    up_exposure = False
    down_exposure = False

    default_facs = False
    fac_mse = 0
    fac_l1 = 0
    fac_ssim = 0
    fac_ms_ssim = 0
    fac_color = 0
    fac_vgg = 0
    fac_texture = 0
    fac_fourier = 0
    fac_frequency = 0
    fac_lpips = 0
    fac_huber = 0
    fac_unet = 0

    for args in arguments:
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]

        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]

        # --- architecture ---
        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("inst_norm"):
            inst_norm = eval(args.split("=")[1])
        
        if args.startswith("num_maps_base"):
            num_maps_base = int(args.split("=")[1])

        # --- model weights ---
        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("patch_w"):
            patch_w = int(args.split("=")[1])

        if args.startswith("patch_h"):
            patch_h = int(args.split("=")[1])

        # --- training options ---
        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # --- more options ---
        if args.startswith("save_mid_imgs"):
            save_mid_imgs = eval(args.split("=")[1])
        if args.startswith("leaky"):
            leaky = eval(args.split("=")[1])
        if args.startswith("norm_gen"):
            norm_gen = args.split("=")[1]
        if args.startswith("norm_disc"):
            norm_disc = args.split("=")[1]
        if args.startswith("flat"):
            flat = int(args.split("=")[1])
        if args.startswith("percentage"):
            percentage = int(args.split("=")[1])
        if args.startswith("entropy"):
            entropy = args.split("=")[1]
        if args.startswith("mix"):
            mix = int(args.split("=")[1])
        if args.startswith("optimizer"):
            optimizer = args.split("=")[1]

        if args.startswith("fac_mse"):
            fac_mse = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_l1"):
            fac_l1 = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ssim"):
            fac_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ms_ssim"):
            fac_ms_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_color"):
            fac_color = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_vgg"):
            fac_vgg = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_texture"):
            fac_texture = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_fourier"):
            fac_fourier = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_frequency"):
            fac_frequency = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_lpips"):
            fac_lpips = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_huber"):
            fac_huber = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_unet"):
            fac_unet = float(args.split("=")[1])
            default_facs = False

        if args.startswith("triple_exposure"):
            triple_exposure = eval(args.split("=")[1])
        if args.startswith("up_exposure"):
            up_exposure = eval(args.split("=")[1])
        if args.startswith("down_exposure"):
            down_exposure = eval(args.split("=")[1])
        if args.startswith("over_dir"):
            over_dir = args.split("=")[1]
        if args.startswith("under_dir"):
            under_dir = args.split("=")[1]


    if default_facs:
        fac_vgg = 0.5
        fac_mse = 200
        fac_ssim = 2
        fac_color = 200
        fac_texture = 1

    # obtain restore iteration info
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    if restore_iter == 0: # no need to get the last iteration if specified
        restore_iter = get_last_iter(model_dir, "DPED")

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]
    num_train_iters += restore_iter

    print("The following parameters will be applied for training:")
    print("Restore Iteration: " + str(restore_iter))
    print("Flat: " + str(flat))
    print("Generator norm: " + norm_gen)
    print("Discriminator norm: " + norm_disc)
    print("Training data pecentage: " + str(percentage))
    print("Sort training images by entropy: " + entropy)
    print("Mixing number of images: " + str(mix))
    print("Optimizer: " + optimizer)
    print("Batch size: " + str(batch_size))
    print("Training size: " + str(train_size))
    print("Learning rate: " + str(learning_rate))
    print("Training iterations: " + str(num_train_iters))
    print("Evaluation step: " + str(eval_step))
    print("Path to the dataset: " + dataset_dir)
    print("Triple exposure: " + str(triple_exposure))
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to result images: " + result_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Path to RGB data from DSLR: " + dslr_dir)
    print("Path to Raw data from phone: " + phone_dir)
    print("Loss function=" +
        " mse:" + str(fac_mse) +
        " l1:" + str(fac_l1) +
        " ssim:" + str(fac_ssim) +
        " ms-ssim:" + str(fac_ms_ssim) +
        " color:" + str(fac_color) +
        " vgg:" + str(fac_vgg) +
        " texture:" + str(fac_texture) +
        " fourier:" + str(fac_fourier) + 
        " frequency:" + str(fac_frequency) +
        " lpips:" + str(fac_lpips) +
        " huber:" + str(fac_huber) +
        " unet:" + str(fac_unet) )
    return dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
        triple_exposure, up_exposure, down_exposure, over_dir, under_dir,\
        patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, \
        save_mid_imgs, leaky, norm_gen, norm_disc, flat, percentage, entropy, mix, optimizer,\
        fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_color, fac_vgg, fac_texture, fac_fourier, fac_frequency, fac_lpips, fac_huber, fac_unet


def process_test_model_args(arguments):

    # Specifying the default parameters for testing

    # --- data path ---
    dataset_dir = 'raw_images/'
    test_dir = 'fujifilm_full_resolution/'
    model_dir = 'models/'
    result_dir = None

    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'

    triple_exposure = False
    up_exposure = False
    down_exposure = False

    # --- architecture ---
    arch = "resnet"
    level = 0
    norm_gen = 'instance'
    num_maps_base = 16
    # --- model weights ---
    orig_model = False
    rand_param = False
    restore_iter = 0
    # --- input size ---
    img_h = 1500 # default size
    img_w = 2000 # default size
    # --- more options ---
    use_gpu = False
    save_model = False
    test_image = True
    flat = 0

    for args in arguments:
        
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("test_dir"):
            test_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]
        
        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]

        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]

        if args.startswith("over_dir"):
            over_dir = args.split("=")[1]

        if args.startswith("under_dir"):
            under_dir = args.split("=")[1]


        # --- architecture ---
        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("level"):
            level = 0 if arch == "pynet_0" else int(args.split("=")[1])

        if args.startswith("norm_gen"):
            norm_gen = args.split("=")[1]

        if args.startswith("num_maps_base"):
            num_maps_base = int(args.split("=")[1])

        # --- model weights ---
        if args.startswith("orig"):
            orig_model = eval(args.split("=")[1])

        if args.startswith("rand"):
            rand_param = eval(args.split("=")[1])

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("img_h"):
            img_h = int(args.split("=")[1])

        if args.startswith("img_w"):
            img_w = int(args.split("=")[1])
        
        # --- more options ---        
        if args.startswith("use_gpu"):
            use_gpu = eval(args.split("=")[1])

        if args.startswith("save"):
            save_model = eval(args.split("=")[1])

        if args.startswith("test_image"):
            test_image = eval(args.split("=")[1])

        if args.startswith("flat"):
            flat = int(args.split("=")[1])

    # choose architecture
    if arch == "resnet":
        name_model = "resnet"

    if result_dir is None:
        result_dir = model_dir

    # obtain restore iteration info (necessary if no pre-trained model or not random weights)
    if restore_iter is None and not orig_model and not rand_param: # need to restore a model

        restore_iter = get_last_iter(model_dir, name_model)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for " + name_model + ".")
            sys.exit()

    print("The following parameters will be applied for testing:")

    print("Model architecture: " + arch)
    print("Flat: " + str(flat))
    print("Restore Iteration: " + str(restore_iter))
    print("Path to the dataset: " + dataset_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to result images: " + result_dir)
    print("Path to testing data: " + test_dir)


    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Triple exposure: " + str(triple_exposure))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)

    return dataset_dir, test_dir, model_dir, result_dir,\
        dslr_dir, phone_dir, over_dir, under_dir, triple_exposure, up_exposure, down_exposure,\
    return dataset_dir, test_dir, model_dir, result_dir,\
        arch, level, norm_gen, num_maps_base, flat, orig_model, rand_param, restore_iter,\
            img_h, img_w, use_gpu, save_model, test_image


def get_last_iter(model_dir, name_model):

    saved_models = [int(model_file.split(".")[0].split("_")[-1])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith(name_model)]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return 0


def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d for d in tensor.get_shape()[1:]), 1)


def export_pb(sess, output_node_name, output_dir='.', export_pb_name='test.pb'):
    gd = sess.graph.as_graph_def()

    # replace variables in a graph with constants for exporting pb
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, gd, [output_node_name]
    )

    tf.io.write_graph(output_graph_def, output_dir, export_pb_name, as_text=False)
    print(f'Save {export_pb_name} in {output_dir} Done')