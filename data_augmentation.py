import numpy as np
import tifffile
from tqdm import tqdm
import random
import scipy 
import tensorflow_addons as tfa
import tensorflow as tf
from tifffile import imwrite
def crop_img_mask(img, mask, chunkh, chunkw, chunkl, imgh, imgw, imgl):
        start_h = random.randint(0, imgh-chunkh)
        start_w = random.randint(0, imgw-chunkw)
        start_l = random.randint(0, imgl-chunkl)
        
        cropped_img = img[start_h:start_h+chunkh, start_w:start_w+chunkw, start_l:start_l+chunkl]
        cropped_mask = mask[start_h:start_h+chunkh, start_w:start_w+chunkw, start_l:start_l+chunkl]
        
        if (cropped_mask.shape == (chunkh, chunkw, chunkl)) & (cropped_img.shape == (chunkh, chunkw, chunkl)):                                       
            return cropped_img , cropped_mask
def rotation(cropped_img, cropped_mask):
    cropped_img = tfa.image.rotate(cropped_img, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
    cropped_mask = tfa.image.rotate(cropped_mask, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
    return cropped_img, cropped_mask
def flip_img(cropped_img, cropped_mask): 
    cropped_img = tf.image.flip_left_right(cropped_img)
    cropped_mask = tf.image.flip_left_right(cropped_mask)
    cropped_img = tf.image.flip_up_down(cropped_img)
    cropped_mask = tf.image.flip_up_down(cropped_mask) 
    return cropped_img, cropped_mask
def random_augmentation(img, mask, chunkh, chunkw, chunkl, imgh, imgw, imgl):
    
    cropped_img, cropped_mask = crop_img_mask(img, mask, chunkh, chunkw, chunkl, imgh, imgw, imgl)

    random_number = random.randint(1,4) 
    if (cropped_mask.shape == (chunkh, chunkw, chunkl)) & (cropped_img.shape == (chunkh, chunkw, chunkl)):
        if random_number == 1:
            cropped_mask = np.array(cropped_mask, dtype=np.bool)
            return cropped_img, cropped_mask
        elif random_number == 2:
            cropped_img = tfa.image.rotate(cropped_img, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
            cropped_mask = tfa.image.rotate(cropped_mask, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
            cropped_img = np.array(cropped_img)
            cropped_mask = np.array(cropped_mask, np.bool)
            return cropped_img, cropped_mask
        elif random_number == 3:
            cropped_img = tf.image.flip_left_right(cropped_img)
            cropped_mask = tf.image.flip_left_right(cropped_mask)
            cropped_img = tf.image.flip_up_down(cropped_img)
            cropped_mask = tf.image.flip_up_down(cropped_mask) 
            cropped_img = np.array(cropped_img)
            cropped_mask = np.array(cropped_mask, np.bool)
            return cropped_img, cropped_mask
        else:
            cropped_img = tfa.image.rotate(cropped_img, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
            cropped_mask = tfa.image.rotate(cropped_mask, tf.constant((2*np.pi)/2), interpolation="BILINEAR")
            cropped_img = tf.image.flip_left_right(cropped_img)
            cropped_mask = tf.image.flip_left_right(cropped_mask)
            cropped_img = tf.image.flip_up_down(cropped_img)
            cropped_mask = tf.image.flip_up_down(cropped_mask) 
            cropped_img = np.array(cropped_img)
            cropped_mask = np.array(cropped_mask, np.bool)
            return cropped_img, cropped_mask
def add_noise(img, nmean=10, nstd=10):
    # generate random Gaussian values
    from numpy.random import seed
    from numpy.random import randn
    # seed random number generator
    seed(1)
    sz = img.shape;
#     nmean = 10; # mean in Gaussian distribution
#     nstd = 10; # std in Gaussian distribution        
    noise_image = randn(sz[0], sz[1], sz[2]) * nstd + nmean;
    raw_image_with_noise = img + noise_image;
    raw_image_with_noise = raw_image_with_noise * (img > 0); # only add noise to the location with intensity.
    return raw_image_with_noise

def add_noise_on_crop(cropped_img, noise_level):
    if noise_level == 1: 
        cropped_img = add_noise(cropped_img, nmean=30 , nstd=20)
        return cropped_img
    elif noise_level == 2: 
        cropped_img = add_noise(cropped_img, nmean=35 , nstd=20)
        return cropped_img
    elif noise_level == 3: 
        cropped_img = add_noise(cropped_img, nmean=40 , nstd=20)
        return cropped_img
    elif noise_level == 4: 
        cropped_img = add_noise(cropped_img, nmean=30 , nstd=24)
        return cropped_img
    elif noise_level == 5: 
        cropped_img = add_noise(cropped_img, nmean=35 , nstd=24)
        return cropped_img
    elif noise_level == 6: 
        cropped_img = add_noise(cropped_img, nmean=40 , nstd=24)
        return cropped_img
    elif noise_level == 7:
        cropped_img = add_noise(cropped_img, nmean=30 , nstd=28)
        return cropped_img
    elif noise_level == 8: 
        cropped_img = add_noise(cropped_img, nmean=35 , nstd=28)
        return cropped_img
    elif noise_level == 9: 
        cropped_img = add_noise(cropped_img, nmean=40 , nstd =28)
        return cropped_img
    else:
        print('Noise Level {}'.format(noise_level) + ' is out of bounds. Please use a number between 1-9 inclusive.')

counter = 0
while counter < 300: 
    cropped_img, cropped_mask = random_augmentation(img1, mask1, 64, 128, 128, 190, 656, 1010)
    tifffile.imwrite('file_path_to_cropped_imgs_' + f"{counter:03}" + '.tif', cropped_img)
    tifffile.imwrite('file_path_to_cropped_masks_' + f"{counter:03}" + '.tif', cropped_mask)
    counter +=1 
    
counter = 300
img5_n = add_noise(img5, nmean=25, nstd=10)
while counter < 600: 
    cropped_img, cropped_mask = random_augmentation(img5_n, mask5, 64, 128, 128, 267, 658, 885)
    tifffile.imwrite('file_path_to_cropped_noisy_imgs_' + f"{counter:03}" + '.tif', cropped_img)
    tifffile.imwrite('file_path_to_cropped_noisy_masks_' + f"{counter:03}" + '.tif', cropped_mask)
    counter +=1
