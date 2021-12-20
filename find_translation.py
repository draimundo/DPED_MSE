from skimage.registration import phase_cross_correlation
from skimage.color import rgb2gray
from scipy.ndimage import shift
from skimage.exposure import equalize_hist
import imageio
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

dataset_dir = 'raw_images/'
type_directory = 'test/'
directory_dslr = dataset_dir + type_directory + 'fujifilm/'
directory_dng = dataset_dir + type_directory + 'mediatek_dng/'

directory_phone = dataset_dir + type_directory + 'mediatek_raw/'
directory_over = dataset_dir + type_directory + 'mediatek_raw_over/'
directory_under = dataset_dir + type_directory + 'mediatek_raw_under/'

shift_directory_over = dataset_dir + type_directory + 'mediatek_raw_shiftover/'
shift_directory_under = dataset_dir + type_directory + 'mediatek_raw_shiftunder/'

TRAINING_IMAGES = [name for name in os.listdir(directory_phone)
                    if os.path.isfile(os.path.join(directory_phone, name))]
TRAINING_IMAGES.sort()

if not os.path.isdir(shift_directory_over):
    os.makedirs(shift_directory_over, exist_ok=True)

if not os.path.isdir(shift_directory_under):
    os.makedirs(shift_directory_under, exist_ok=True)

for img in tqdm(TRAINING_IMAGES, miniters=100):
    dslr = np.asarray(imageio.imread(directory_dslr + str(img)))
    dng = np.asarray(imageio.imread(directory_dng + str(img)))
    # orig = np.asarray(imageio.imread(directory_phone + str(img)))
    # over = np.asarray(imageio.imread(directory_over + str(img)))
    # under = np.asarray(imageio.imread(directory_under + str(img)))

    # shiftover, error, diffphase = phase_cross_correlation(equalize_hist(orig), equalize_hist(over), upsample_factor=100)
    # shiftunder, error, diffphase = phase_cross_correlation(equalize_hist(orig), equalize_hist(under), upsample_factor=100)
    corrdng, error, diffphase = phase_cross_correlation(equalize_hist(rgb2gray(dslr)), equalize_hist(rgb2gray(dng)), upsample_factor=100)

    # corrover = 2*np.round(shiftover/2)
    # corrunder = 2*np.round(shiftunder/2)

    # if np.max(np.abs(corrover)):
    #     # print("Detected subpixel offset over (y, x): {}".format(shiftover))
    #     # print("Correction over (y, x): {}".format(corrover))
    #     fixover = shift(over, corrover, order=0, mode='mirror')
    #     # f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
    #     # ax1.imshow(equalize_hist(orig))
    #     # ax2.imshow(equalize_hist(over))
    #     # ax3.imshow(equalize_hist(fixover))
    #     # plt.show()
    #     imageio.imwrite(shift_directory_over + img, fixover)

    # if np.max(np.abs(corrunder)):
    #     # print("Detected subpixel offset under (y, x): {}".format(shiftunder))
    #     # print("Correction under (y, x): {}".format(corrunder))
    #     fixunder = shift(under, corrunder, order=0, mode='mirror')
    #     # f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
    #     # ax1.imshow(equalize_hist(orig))
    #     # ax2.imshow(equalize_hist(under))
    #     # ax3.imshow(equalize_hist(fixunder))
    #     # plt.show()
    #     imageio.imwrite(shift_directory_under + img, fixunder)

    if np.max(np.abs(corrdng))>3:
        print("Correction under (y, x): {}".format(corrdng))
        fixdng = shift(dng, np.insert(corrdng, 0, 0), order=0, mode='mirror')
        f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
        ax1.imshow(dslr)
        ax2.imshow(dng)
        ax3.imshow(fixdng)
        plt.show()
        # imageio.imwrite(shift_directory_under + img, fixunder)