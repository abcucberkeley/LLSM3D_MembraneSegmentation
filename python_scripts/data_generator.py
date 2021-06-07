"""

LLSM 3D Membrane Segmentation using 3D U-Net
Author: Zeeshan Patel
Instructions: Before running this data generation script, please check all comments 
              in the last three sections of the script (Data Loading, Data Generation, Saving Data) 
              and configure the script to your needs.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
import sys
import numpy as np 
import tifffile
import random
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import math
from numpy.random import randn


# Data Generation Parameters 

np.random.seed(42)

IMG_WIDTH = 112
IMG_HEIGHT = 112
IMG_DEPTH = 32
IMG_CHANNELS = 1
crops_per_img = 225 # Configure as needed.

##########################################################################################################################
##########################################################################################################################

# Data Augmentation Functions

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

def add_noise(img, nmean=10, nstd=10):
    sz = img.shape;
#     nmean = 10; # mean in Gaussian distribution
#     nstd = 10; # std in Gaussian distribution        
    noise_image = randn(sz[0], sz[1], sz[2]) * nstd + nmean;
    raw_image_with_noise = img + noise_image;
    raw_image_with_noise = raw_image_with_noise * (img > 0); # only add noise to the location with intensity.
    return raw_image_with_noise

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
            cropped_img = add_noise(cropped_img, nmean=10, nstd=10)
            cropped_mask = np.array(cropped_mask, np.bool)
            return cropped_img, cropped_mask
        elif random_number == 3:
            cropped_img = tf.image.flip_left_right(cropped_img)
            cropped_mask = tf.image.flip_left_right(cropped_mask)
            cropped_img = tf.image.flip_up_down(cropped_img)
            cropped_mask = tf.image.flip_up_down(cropped_mask) 
            cropped_img = np.array(cropped_img)
            cropped_img = add_noise(cropped_img, nmean=10, nstd=10)
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


##########################################################################################################################
##########################################################################################################################

# Loading Your Normalized Training Images and Binarized Training Masks


img = tifffile.imread('/some/path/normalized_image.tif') # Add more images below as needed.


print('img: ' + f"{img.shape}") # Add more images as needed. 

mask = tifffile.imread('/some/path/binarized_mask.tif') # Add more masks below as needed.

# Converting masks from bool to uint8 in order to use augmentation functions. Add more masks below as needed.
mask.dtype = np.uint8


print('\n Mask: ' + f"{mask.shape}") # Add more masks as needed.

##########################################################################################################################
##########################################################################################################################

# Generating Cropped Training Images and Masks
        
x = []
y = []
count = 0
while count < crops_per_img:
    
    cropped_img1, cropped_mask1 = random_augmentation(img1, mask1, 32, 112, 112, 171, 656, 1008) # Configure dimensions as needed.
    x.append(cropped_img1)
    y.append(cropped_mask1)
    
    # If you have more images to generate data from, you may add them below using the same format as the example cropping code.

    count+=1
    
##########################################################################################################################
##########################################################################################################################

# Saving Cropped Training Images and Masks
    
for i in tqdm(range(len(x))):
    tifffile.imwrite('/some/path/cropped_imgs/img{}.tif'.format(i), x[i]) # Please configure your filepath for your training images. 
    tifffile.imwrite('/some/path/cropped_masks/mask{}.tif'.format(i), y[i]) # Please configure your filepath for your training masks. 
    
