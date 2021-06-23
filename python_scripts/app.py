"""

LLSM 3D Membrane Segmentation using 3D U-Net
Author: Zeeshan Patel
Instructions: This app was created using Streamlit. Follow the instructions on the official documentation to learn how to deploy/use this app.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import streamlit as st
import numpy as np
import math
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import tensorflow_addons as tfa
import tensorflow as tf
from tifffile import imwrite
from numpy.random import randn
import skimage 
from skimage import morphology
import json
import os
import sys
import random
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import datetime
from skimage.morphology import disk, local_minima
from skimage.filters import rank
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage import measure
import scipy.ndimage as ndi
import scipy
import time 



add_selectbox = st.sidebar.selectbox(
    "Navigate to a webpage",
    ("Home", "Data Preparation", "Data Generation", 
    "3D U-Net Training", "Inference", "Segmentation")
)

if add_selectbox == "Home":

    st.title("LLSM 3D Membrane Segmentation", )

    st.title("Introduction")

    st.write("Adaptive Optical Lattice light-sheet microscopy (AO-LLSM) is a 3D live-cell imaging method that reduces the damage to the physiological state of the biological specimen due to phototoxicity while also maintaining high-resolution image quality by avoiding issues with problems with aberrations (Gao et al., 2019; Liu et al., 2018). AO-LLSM delivers images in high spatiotemporal resolution which allows for detailed analysis of individual cells in complex, multicellular organisms. However, identifying cell boundaries in dense, multicellular organisms can be a daunting and tedious task. In order to understand cellular processes, it is vital to properly identify cell membranes and to label these individual cell membranes. Precise labeling will enable the isolation of any cell in images of dense, multicellular organisms and analyze its cellular dynamics, cell morphologies, and organelle processes. We outline a 3D cell membrane segmentation method using deep learning with 3D U-Net. We developed our own image normalization and ground truth label binarization algorithms for data preprocessing using core frameworks such as NumPy and scikit-image. To generate training and testing datasets, we created random augmentation algorithms that utilized noise functions and TensorFlow based image augmentation functions. To develop the 3D U-Net, we utilized Tensorflow-based Keras and used the original architecture (Özgün Çiçek et al., 2016). To use our tool on large, volumetric datasets, we developed a prediction script that can take in 3D cell membrane images of any size and produce a predicted label for the image. Our 3D U-Net was able to generate labels with the top accuracy of 96.55% in less than five hours. This process allows fast image processing and rapid neural network training, providing a both time and cost-efficient automated method for 3D cell membrane detection and segmentation.")

    st.title("Documentation")

    docs_link = '[LLSM 3D Membrane Segmentation Docs](https://docs.google.com/document/d/1pYugyeuD_Ypg7Xc-aFosiLLn-595bRr3oNI_ojKiK1A/edit?usp=sharing)'
    st.markdown(docs_link, unsafe_allow_html=True)

    st.title("Github")

    github_link = '[LLSM 3D Membrane Segmentation Repository](https://github.com/abcucberkeley/LLSM3D_MembraneSegmentation)'
    st.markdown(github_link, unsafe_allow_html=True)

    st.title("References")

    st.write("(1) Özgün Çiçek et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. arXiv:1606.06650. (2) Gao et al. (2019). Cortical column and whole-brain imaging with molecular contrast and nanoscale resolution. Science Vol. 363, Issue 6424. (3) Liu et al. (2018). Observing the cell in its native state: Imaging subcellular dynamics in multicellular organisms. Science Vol. 360, Issue 6386. (4) Aguet et al. (2016) Membrane dynamics of dividing cells imaged by lattice light-sheet microscopy. Molecular Biology of the Cell 2016 27:22, 3418-3435.")

    st.title("Citation")

    st.write("*fill text here*")


if add_selectbox == "Data Preparation":

    st.title("Data Preparation")
    st.write("Processing raw data before conducting a neural network training is essential. On this page, you can pre-process your raw images and masks to ensure your neural network generates accurate segmentations.")
    st.write("Data preparation entails two processes: mask binarization and raw image normalization. **Mask Binarization** can require many hours depending on the size of the mask. **Raw Image Normalization** usually takes a few minutes and is much faster to complete.")

    st.write('')

    select_dataprep = st.selectbox('Choose a data preparation task.', ["Raw Image Normalization", "Mask Binarization"])


    if select_dataprep == "Raw Image Normalization":
        st.title("Raw Image Normalization")
        st.write('')
        path = st.text_input("Enter the file path to your raw image.")

        if path:
            raw_img = tifffile.imread(path)

            save_path = st.text_input("Enter the file path to save the scaled image.")

            ranges = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 95.1, 95.2, 95.3, 95.4, 95.5, 95.6, 95.7, 95.8, 95.9, 96. , 96.1,
            96.2, 96.3, 96.4, 96.5, 96.6, 96.7, 96.8, 96.9, 97. , 97.1, 97.2,
            97.3, 97.4, 97.5, 97.6, 97.7, 97.8, 97.9, 98. , 98.1, 98.2, 98.3,
            98.4, 98.5, 98.6, 98.7, 98.8, 98.9, 99.,  99.05,  99.1 ,  99.15,  99.2 ,  99.25,  99.3 ,  99.35,  99.4 ,
                99.45,  99.5 ,  99.55,  99.6 ,  99.65,  99.7 ,  99.75,  99.8 ,
                99.85,  99.9 ,  99.95, 100.  ])
            
            img = np.ndarray.flatten(raw_image)
            intensity = np.zeros(shape=(155,))
            for i in range(len(ranges)): 
                intensity_at_percentile = np.percentile(raw_image, ranges[i])
                intensity[i] = intensity_at_percentile
            
            def scaleContrast(raw_img, rangeIn=(0,800), rangeOut=(0,255)):
                x = rangeOut[1] - rangeIn[0]
                if x == 0:
                    out = np.zeros(shape=raw_img.shape)
                else: 
                    out = (raw_img-rangeIn[0])/np.diff(rangeIn) * np.diff(rangeOut) + rangeOut[0]
                
                for i in tqdm(range(len(out))):
                    for j in range(len(out[i])):
                        for k in range(len(out[i][j])):
                            if out[i][j][k] > rangeOut[1]:
                                out[i][j][k] = rangeOut[1]
                return np.uint8(out)
            
            img_scaled = scaleContrast(raw_image, rangeIn=(0, intensity[-2]), rangeOut=(0,255))

            tifffile.imwrite('{}'.format(save_path), img_scaled)

            st.write("Normalized Image Saved Successfully!")

    elif select_dataprep == "Mask Binarization":
        st.title("Mask Binarization")
        st.write('')
        path = st.text_input("Enter the file path to your mask.")

        if path:
            mask = tifffile.imread(path)
            save_path = st.text_input("Enter the file path to save the binary mask.")
            arr = np.unique(mask)
            arr = np.delete(arr, np.where(arr == 0))
            arr = np.delete(arr, np.where(arr == 1))

            def binarize(mask, arr): # arr is an array with all unique values of mask except 0 and 1
                mask_bw = 0 * mask # creating new array of the same size as mask, except all 0s
                for i in tqdm(range(len(arr))): 
                    bw_i = (mask == arr[i]) # setting bw_i equal to every instance of a unique pixel value in mask
                    bw_i = skimage.morphology.erosion(bw_i) # performing erosion
                    mask_bw = mask_bw + bw_i 
                return mask_bw

            mask_bm = binarize(mask, arr)

            def rm_cytosol(mask, binary_mask):
                b_mask = mask>1 # mask --> original mask
                final_mask = binary_mask - b_mask
                return final_mask
            
            def conv_to_boolean(final_mask):
                final_mask.dtype = np.bool 
                final_mask = final_mask[::, ::, 1::2]
                return final_mask
            
            final_mask = rm_cytosol(mask, mask_bm)
            final_mask = conv_to_boolean(final_mask)
            tifffile.imwrite(save_path, final_mask)

            st.write("Binarized Mask Saved Successfully!")


elif add_selectbox == "Data Generation":
    
    st.title("Data Generation")
    st.write('')
    st.write("The data generator allows you to create large augmented datasets for 3D U-Net training. These large datasets are obtained by cropping and then rotating, flipping, and adding Gaussian noise to the original images and their masks. In our experimentation, we utilized specific parameters for our dataset to optimize our neural network with the available computational resources. You can read more about our dataset specs below.")

    st.subheader('Recommended Dataset Specifications')
    st.markdown("""
                #### Cropped image dimensions
                - IMG_WIDTH = 112
                - IMG_HEIGHT = 112
                - IMG_DEPTH = 32
                - Gaussian Noise Parameters: µ = 10, σ = 10
                - 2500 images used in total for 3 GPU training
                """)
    
    st.write('')
    select_datagen = st.selectbox("Would you like to use our recommended data specifications or custom ones?", ["Reccomended Specifications", "Custom Specifications"])
    
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

    def add_noise(img, nmean=10, nstd=10):     # nmean = 10; # mean in Gaussian distribution  # nstd = 10; # std in Gaussian distribution  
        sz = img.shape      
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


    if select_datagen == "Reccomended Specifications":
        num_imgs = st.slider("How many full images are you using to generate your training dataset?", 1, 15)

        raw_img_paths = []
        mask_paths = []
        for i in range(num_imgs):
            raw_img_path = st.text_input("Enter the path to raw image {}".format(i+1))
            mask_path = st.text_input("Enter the path to mask {}".format(i+1))
            if raw_img_path != None and mask_path != None:
                raw_img_paths.append(raw_img_path)
                mask_paths.append(mask_path)
        
        if len(raw_img_paths) > 1 and len(mask_paths) > 1:

            save_path_imgs = st.text_input("Enter the path to the folder you would like to save your raw image crops in.")
            save_path_masks = st.text_input("Enter the path to the folder you would like to save your mask crops in.")
            
            if save_path_imgs and save_path_masks:
                imgs = []
                masks = []

                for i in range(len(raw_img_paths)):
                    x = tifffile.imread(raw_img_paths[i])
                    y = tifffile.imread(mask_paths[i])
                    y.dtype = np.uint8
                    if x.shape == y.shape:
                        imgs.append(x)
                        masks.append(y)
                    else:
                        error = True
                        st.write("Error: Image {} and Mask {} have different shapes.")
                
                
                if error != True:
                    num_crops = (2500/num_imgs)

                    x = []
                    y = []
                    for i in range(len(imgs)):
                        if i < len(imgs) - 1:
                            for j in range(num_crops):
                                cropped_img, cropped_mask = random_augmentation(imgs[i], masks[i], 32, 112, 112, img.shape[0], img.shape[1], img.shape[2])
                                x.append(cropped_img)
                                y.append(cropped_mask)
                        else:
                            for j in range(num_crops + (2500 % num_imgs)):
                                cropped_img, cropped_mask = random_augmentation(imgs[i], masks[i], 32, 112, 112, img.shape[0], img.shape[1], img.shape[2])
                                x.append(cropped_img)
                                y.append(cropped_mask)
                    
                    for i in tqdm(range(len(x))):
                        tifffile.imwrite(save_path_imgs + 'img{}.tif'.format(i), x[i])
                        tifffile.imwrite(save_path_imgs + 'mask{}.tif'.format(i), y[i])
                    
                    st.write('Data Generation Complete!')

    elif select_datagen == "Custom Specifications":
        num_imgs = st.slider("How many full images are you using to generate your training dataset?", 1, 15)
        st.write(' ')
        cropped_num = st.text_input("How many cropped images would you like to generate?")
        st.write(' ')
        st.write('The cropped image dimensions should not be too large as it can cause too much memory to be used, which will lead to resource errors during training.')
        st.write(' ')
        IMG_WIDTH = st.text_input("Please enter a cropped image width for your data. Ensure that this number is divisible by 8.")
        st.write(' ')
        IMG_HEIGHT = st.text_input("Please enter a cropped image height for your data. Ensure that this number is divisible by 8.")
        st.write(' ')
        IMG_DEPTH = st.text_input("Please enter a cropped image depth for your data. Ensure that this number is divisible by 8.")
        st.write(' ')
        if IMG_WIDTH and IMG_HEIGHT and IMG_DEPTH and cropped_num:
            IMG_WIDTH = int(IMG_WIDTH)
            IMG_HEIGHT = int(IMG_HEIGHT)
            IMG_DEPTH = int(IMG_DEPTH)
            cropped_num = int(cropped_num)
        raw_img_paths = []
        mask_paths = []
        for i in range(num_imgs):
            raw_img_path = st.text_input("Enter the path to raw image {}".format(i+1))
            mask_path = st.text_input("Enter the path to mask {}".format(i+1))
            if raw_img_path != None and mask_path != None:
                raw_img_paths.append(raw_img_path)
                mask_paths.append(mask_path)
        
        if len(raw_img_paths) > 1 and len(mask_paths) > 1:

            save_path_imgs = st.text_input("Enter the path to the folder you would like to save your raw image crops in. (Ex: /my/computer/cropped_imgs/)")
            save_path_masks = st.text_input("Enter the path to the folder you would like to save your mask crops in. (Ex: /my/computer/cropped_masks/)")
            
            if save_path_imgs and save_path_masks:
                imgs = []
                masks = []

                for i in range(len(raw_img_paths)):
                    x = tifffile.imread(raw_img_paths[i])
                    y = tifffile.imread(mask_paths[i])
                    y.dtype = np.uint8
                    if x.shape == y.shape:
                        imgs.append(x)
                        masks.append(y)
                    else:
                        error = True
                        st.write("Error: Image {} and Mask {} have different shapes.")
                
                
                if error != True:
                    num_crops = (cropped_num/num_imgs)

                    x = []
                    y = []
                    for i in range(len(imgs)):
                        if i < len(imgs) - 1:
                            for j in range(num_crops):
                                cropped_img, cropped_mask = random_augmentation(imgs[i], masks[i], IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, img.shape[0], img.shape[1], img.shape[2])
                                x.append(cropped_img)
                                y.append(cropped_mask)
                        else:
                            for j in range(num_crops + (cropped_num % num_imgs)):
                                cropped_img, cropped_mask = random_augmentation(imgs[i], masks[i], IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, img.shape[0], img.shape[1], img.shape[2])
                                x.append(cropped_img)
                                y.append(cropped_mask)
                    
                    for i in tqdm(range(len(x))):
                        tifffile.imwrite(save_path_imgs + 'img{}.tif'.format(i), x[i])
                        tifffile.imwrite(save_path_imgs + 'mask{}.tif'.format(i), y[i])
                    
                    st.write('Data Generation Complete!')
        

elif add_selectbox == "3D U-Net Training":

    seed = 42
    np.random.seed = seed

    st.title("3D U-Net Training")
    st.write(" ")
    st.write("Here, you can use your generated data to train a 3D U-Net model to detect and classify cell membranes. You can view the live progress of your model training through your terminal/command prompt.")
    st.write(" ")
    st.write("If you are not interested in training your own model, you can go to the inference page using the sidebar to use our pretrained model.")
    st.write(" ")
    st.write("For this process to work successfully, you must utilize our data generator to create your training dataset. Additionally, ensure that you are on a GPU cluster node and have access to more than one GPU. It is recommended to use 3 GPUs for the training.")
    st.write(" ")
    st.write("Number of GPUs Currently Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    st.write(" ")
    st.write("Any logs for Tensorboard will be saved to ../logs/fit.")
    st.write(" ")
    cropped_num = st.text_input("How many cropped images would you like to use in the training? (Only count the number of raw images, not masks.)")
    st.write(' ')
    IMG_WIDTH = st.text_input("Please enter the cropped image width for your data. Ensure that this number is divisible by 8.")
    st.write(' ')
    IMG_HEIGHT = st.text_input("Please enter the cropped image height for your data. Ensure that this number is divisible by 8.")
    st.write(' ')
    IMG_DEPTH = st.text_input("Please enter the cropped image depth for your data. Ensure that this number is divisible by 8.")
    st.write(' ')
    raw_path = st.text_input("Please enter the path to the folder where your cropped raw images are stored. (Ex: /my/computer/cropped_imgs/)")
    st.write(' ')
    mask_path = st.text_input("Please enter the path to the folder where your cropped masks are stored. (Ex: /my/computer/cropped_masks/)")
    st.write(' ')
    epoch_selection = st.text_input("How many epochs would you like to train the model for?")
    st.write(' ')
    batch_size_selection = st.text_input("What batch size would you like for the model training? (Must be divisible by 8; Recommended to use batch size of 8 to start off.)")
    st.write(' ')
    model_save_path = st.text_input("Where would you like to save your final trained model? Add a .h5 extension for the file name. (Ex: /my/computer/models/myfirstmodel.h5)")
    st.write(' ')
    plot_name = st.text_input("What would you like to name your final accuracy vs. loss plot? (Do not enter a file path. The plot will be saved in the same directory as this app's script.)")
    st.write(' ')
    callback_selection = st.selectbox("Do you want to enable the Tensorboard and/or ModelCheckpoint callbacks?", ["Yes", "No"])
    st.write(' ')
    callbacks = []
    if callback_selection == "Yes":
        tensorboard = st.selectbox("Do you want to enable the Tensorboard callback?", ["Yes", "No"])
        st.write(' ')
        modelcheckpoint = st.selectbox("Do you want to enable the ModelCheckpoint callback?", ["Yes", "No"])
        st.write(' ')
        if tensorboard == "Yes":
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs/fit', histogram_freq=1))
        if modelcheckpoint == "Yes":
            filepath = st.text_input("Enter a filepath to save your checkpoints. (Ex: /my/computer/model1)")
            monitor_selection = st.selectbox("Which metric would you like to monitor for the ModelCheckpoint?", ["val_accuracy", "val_loss", "accuracy", "loss"])
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor=monitor_selection, save_best_only=True))
    if IMG_WIDTH and IMG_HEIGHT and IMG_DEPTH and cropped_num and epoch_selection:
        IMG_WIDTH = int(IMG_WIDTH)
        IMG_HEIGHT = int(IMG_HEIGHT)
        IMG_DEPTH = int(IMG_DEPTH)
        IMG_CHANNELS = 1
        cropped_num = int(cropped_num)
        epoch_selection = int(epoch_selection)
        batch_size_selection =  int(batch_size_selection)
        x = []
        y = []
        count = 0
        if raw_path and mask_path and cropped_num:
            while count < int(cropped_num):
                img = tifffile.imread(raw_path + 'img{}.tif'.format(count))
                x.append(img)
                mask = tifffile.imread(mask_path + 'mask{}.tif'.format(count))
                y.append(mask)
                count+=1
        test_size = st.slider("What proportion of your training data would you like to use for validation?", 0.01, 1.00)
        if test_size and x and y:
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
            x_train = np.array(x_train, dtype=np.uint8)
            y_train = np.array(y_train, dtype=np.bool)
            x_test = np.array(x_test, dtype=np.uint8)
            y_test = np.array(y_test, dtype=np.bool)

            st.markdown('#### Data Loaded Successfully!')
            st.balloons()

            def dice_coef(y_true, y_pred, smooth=1e-6):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = K.sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            def dice_loss(y_true, y_pred):
                smooth = 1.
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = y_true_f * y_pred_f
                score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
                return 1. - score

            def bce_dice_loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                inputs = tf.keras.layers.Input((IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
                c1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(s)
                c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
                c1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c1)
                c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
                p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

                c2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(p1)
                c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
                c2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c2)
                c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
                p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)

                c3 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(p2)
                c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
                c3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c3)
                c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
                p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3)

                c4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(p3)
                c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)
                c4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c4)
                c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)

                u6 = tf.keras.layers.Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer='l2')(c4)
                u6 = tf.keras.layers.concatenate([u6, c3])
                c6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(u6)
                c6 = tf.keras.layers.BatchNormalization(axis=-1)(c6)
                c6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c6)
                c6 = tf.keras.layers.BatchNormalization(axis=-1)(c6)

                u7 = tf.keras.layers.Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer='l2')(c6)
                u7 = tf.keras.layers.concatenate([u7, c2])
                c7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(u7)
                c7 = tf.keras.layers.BatchNormalization(axis=-1)(c7)
                c7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c7)
                c7 = tf.keras.layers.BatchNormalization(axis=-1)(c7)

                u8 = tf.keras.layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer='l2')(c7)
                u8 = tf.keras.layers.concatenate([u8, c1])
                c8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(u8)
                c8 = tf.keras.layers.BatchNormalization(axis=-1)(c8)
                c8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer='l2')(c8)
                c8 = tf.keras.layers.BatchNormalization(axis=-1)(c8)
                outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_regularizer='l2')(c8)
                opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)

                model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
                model.compile(optimizer=opt, loss=bce_dice_loss, metrics=['accuracy'])

            model.summary()

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            batch_size = batch_size_selection
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
            test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
            begin_time = datetime.datetime.now()
            u_net = model.fit(train_dataset, epochs=epoch_selection, verbose=1, validation_data=test_dataset)
            model.save(model_save_path)
            st.write("Model Training Complete!")
            st.write(' ')
            st.balloons()
            st.write("Time Taken for Model Training: {}".format(datetime.datetime.now() - begin_time))
            

            plt.style.use('seaborn-darkgrid')
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(50,40))
            t = f.suptitle('3D U-Net Performance', fontsize=90, fontweight='bold')
            f.subplots_adjust(top=0.85, wspace=0.4)

            max_epoch = len(u_net.history['accuracy'])+1
            epoch_list = list(range(1,max_epoch))
            ax1.plot(epoch_list, u_net.history['accuracy'], label='Train Accuracy', color='blue', linewidth=6.5)
            ax1.plot(epoch_list, u_net.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=6.5)
            ax1.set_ylabel('Accuracy Value', fontsize=50, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=50, fontweight='bold')
            ax1.set_title('Accuracy', fontsize=70, fontweight='bold')
            # ax1.set_yscale('log')
            plt.setp(ax1.get_xticklabels(), fontsize=38, fontweight="bold", horizontalalignment="left")
            plt.setp(ax1.get_yticklabels(), fontsize=38, fontweight="bold", horizontalalignment="right")
            l1 = ax1.legend(loc='best', prop={'size': 35})

            ax2.plot(epoch_list, u_net.history['loss'], label='Train Loss', color='blue', linewidth=6.5)
            ax2.plot(epoch_list, u_net.history['val_loss'], label='Validation Loss', color='red', linewidth=6.5)
            ax2.set_ylabel('Loss Value', fontsize=50, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=50, fontweight='bold')
            ax2.set_title('Loss', fontsize=70, fontweight='bold')
            # ax2.set_yscale('log')
            plt.setp(ax2.get_xticklabels(), fontsize=38, fontweight="bold", horizontalalignment="left")
            plt.setp(ax2.get_yticklabels(), fontsize=38, fontweight="bold", horizontalalignment="right")
            l2 = ax2.legend(loc='best', prop={'size': 35})
            plt.savefig(plot_name + ".png", format='png', dpi=500)


elif add_selectbox == "Inference":

    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        smooth = 1.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1. - score

    def bce_dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    model = None
    st.title("Inference")
    st.write(' ')
    st.write("Here, you can either load your own 3D U-Net model or use our pre-trained model for generating a cell boundary inference. The inference algorithm will use the CNN model to predict on subvolumes of the image you input. The prediction subvolumes will be stitched together to form the final full inference.")
    st.write(' ')
    inference_mode = st.selectbox('Would you like to use a pre-trained or custom 3D U-Net model for inference?', ['Pre-Trained', 'Custom'])
    if inference_mode == "Pre-Trained":
        model_exists = os.path.isfile('/clusterfs/fiona/zeeshan/unet_models/unet_test_distributed_13.h5')
        if model_exists == True:
            model = tf.keras.models.load_model('/clusterfs/fiona/zeeshan/unet_models/unet_test_distributed_13.h5', custom_objects={'bce_dice_loss': bce_dice_loss})
        else:
            st.markdown('#### Model is currently unavailable. Please use our custom option.')
    else:
        custom_file_path = st.text_input('Enter the filepath for your custom model.')
        if custom_file_path:
            model = tf.keras.models.load_model(custom_file_path, custom_objects={'bce_dice_loss': bce_dice_loss})

    if model:
        st.write(' ')
        img_path = st.text_input("Enter the filepath to the raw image you would like to generate an inference for. Ensure that the input image is using the tiff file format. (Ex: /my/computer/image/image.tif)")
        save_path = st.text_input("Enter the desired path to save your inference. Ensure that you use the '.tif' file extension. (Ex: /my/computer/predictions/pred.tif)")
        if img_path:
            img = tifffile.imread(img_path)
            st.write(' ')
            if inference_mode == "Custom":
                CHUNK_HEIGHT = st.text_input('Enter the training image depth used to trained your custom model.')
                CHUNK_LENGTH = st.text_input('Enter the training image length used to trained your custom model.')
                CHUNK_WIDTH = st.text_input('Enter the training image width used to trained your custom model.')

                if CHUNK_HEIGHT and CHUNK_LENGTH and CHUNK_WIDTH:
                    CHUNK_HEIGHT = int(CHUNK_HEIGHT)
                    CHUNK_LENGTH = int(CHUNK_LENGTH)
                    CHUNK_WIDTH = int(CHUNK_WIDTH)

            if inference_mode == "Pre-Trained":
                CHUNK_HEIGHT = 32
                CHUNK_LENGTH = 112
                CHUNK_WIDTH = 112
        
            OL = st.slider('How many pixels would you like to use as an overlap? (Must be an even number greater than 0. Recommeneded 10 pixels.)', 2, 30, step=2)
            OL = int(OL)
            def full_volume_prediction(img, imgh, imgl, imgw, chunkh, chunkl, chunkw, OL):
                    if OL%2 !=0:
                        print('ERROR: Please enter an even OL value for the program to run.')
                    else:
                        a = imgh/(chunkh-OL)
                        a = int(math.ceil(a)) + 1
                        b = imgl/(chunkl-OL)
                        b = int(math.ceil(b)) + 1
                        c = imgw/(chunkw-OL)
                        c = int(math.ceil(c))
                        i = 0
                        xmin = np.zeros(shape=(a*b*c,), dtype=np.uint16)
                        xmax = np.zeros(shape=(a*b*c,), dtype=np.uint16)
                        ymin = np.zeros(shape=(a*b*c,), dtype=np.uint16)
                        ymax = np.zeros(shape=(a*b*c,), dtype=np.uint16)
                        zmin = np.zeros(shape=(a*b*c,), dtype=np.uint16)
                        zmax = np.zeros(shape=(a*b*c,), dtype=np.uint16)

                        num_chunks = 0
                        for z in range(a): 
                            for y in range(b):
                                for x in range(c):
                                    xmin[num_chunks] = (((x)*(chunkw-OL)))
                                    if x < chunkw: 
                                        xmax[num_chunks] = (chunkw*(x+1)-(OL*(x)))
                                        if xmax[num_chunks] > imgw:
                                            xmin[num_chunks] = imgw-chunkw
                                            xmax[num_chunks] = imgw

                                    ymin[num_chunks] = (((y)*(chunkl-OL)))
                                    if y < chunkl: 
                                        ymax[num_chunks] = (chunkl*(y+1)-(OL*(y)))
                                        if ymax[num_chunks] > imgl:
                                            ymin[num_chunks] = imgl-chunkl
                                            ymax[num_chunks] = imgl

                                    zmin[num_chunks] = (((z)*(chunkh-OL)))
                                    if z < chunkh: 
                                        zmax[num_chunks] = (chunkh*(z+1)-(OL*(z)))
                                        if zmax[num_chunks] > imgh:
                                            zmin[num_chunks] = imgh-chunkh
                                            zmax[num_chunks] = imgh
                                    num_chunks+=1

                        halfOL = int(OL/2)
                        full_pred = np.zeros(shape=(imgh, imgl, imgw))
                        print(full_pred.shape)
                        while i < (num_chunks): 
                            print(i)
                            print('Zmin:', zmin[i], "Zmax:", zmax[i], 'Ymin:', ymin[i], "Ymax:", ymax[i],'Xmin:', xmin[i], "Xmax:", xmax[i])
                            cropped_img = img[zmin[i]:zmax[i], ymin[i]:ymax[i], xmin[i]:xmax[i]] 
                            print(cropped_img.shape)
                            if (cropped_img.shape == (chunkh, chunkl, chunkw)):
                                pd = model.predict(cropped_img.reshape([1, chunkh, chunkl, chunkw, 1]), verbose=1)
                                #Postprocess pd
                                cropped_pred = pd_preprocessing(pd) 
                                full_pred[zmin[i]+halfOL:zmax[i]-halfOL, ymin[i]+halfOL:ymax[i]-halfOL, xmin[i]+halfOL:xmax[i]-halfOL] = cropped_pred[0+halfOL:chunkh-halfOL, 0+halfOL:chunkl-halfOL, 0+halfOL:chunkw-halfOL]
                            else:
                                print('This crop was too small.')
                                new_array = np.zeros(shape=(chunkh,chunkl,chunkw))
                                arr = np.zeros(shape=cropped_img.shape)
                                xdiff = cropped_img.shape[2] 
                                ydiff = cropped_img.shape[1] 
                                zdiff = cropped_img.shape[0] 
                                print('Zdiff', zdiff, "Ydiff", ydiff, 'Xdiff', xdiff)
                                new_array[0:zdiff, 0:ydiff, 0:xdiff] = cropped_img
                                pd = model.predict(new_array.reshape([1, chunkh, chunkl, chunkw, 1]), verbose=1)
                                cropped_pred = pd_preprocessing(pd)
                                arr = cropped_pred[0:zdiff, 0:ydiff, 0:xdiff]
                                if zdiff>halfOL:
                                    zidx = list(range(halfOL, zdiff-halfOL+1))
                                else:
                                    zidx = list(range(0, zdiff+1))
                                
                                if ydiff>halfOL:
                                    yidx = list(range(halfOL, ydiff-halfOL+1))
                                else:
                                    yidx = list(range(0, ydiff+1))         
                                
                                if xdiff>halfOL:
                                    xidx = list(range(halfOL, xdiff-halfOL+1))
                                else:
                                    xidx = list(range(0, xdiff+1))
                                
                                full_pred[zmin[i]:zmin[i]+len(zidx)-1, ymin[i]:ymin[i]+len(yidx)-1, xmin[i]:xmin[i]+len(xidx)-1] = arr[zidx[0]:zidx[len(zidx)-1], yidx[0]:yidx[len(yidx)-1], xidx[0]:xidx[len(xidx)-1]]
                                
                            i+=1
                                
                        return np.float32(full_pred)

            pred = full_volume_prediction(img, img.shape[0], img.shape[1], img.shape[2], CHUNK_HEIGHT, CHUNK_LENGTH, CHUNK_WIDTH, OL)
            tifffile.imsave(save_path, pred)

            st.write(' ')
            st.markdown('#### Prediction Complete!')
            st.balloons()


elif add_selectbox == "Segmentation":

    st.title("Segmentation")
    st.write(' ')
    st.write("On this page, you can utilize our watershed segmentation feature to generate 3D cell segmentations for your inferences. This process can take 20-30 minutes for large images.")
    
    def watershed_seg(inference_path, threshold, save_path, watershed_line=False):
        start_time = time.time()
        st.write("Starting segmentation at {}".format(start_time))
        inf = tifffile.imread(inference_path)
        st.write("Inference Read\nInference Shape: {}".format(inf.shape))
        mask = inf < threshold
        st.write("Created binary mask")
        distance = ndi.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, labels=mask, footprint=np.ones((3, 3, 3)), indices=False)
        markers = ndi.label(local_maxi)[0]
        watershed_test = segmentation.watershed(inf, mask=mask, markers=markers, watershed_line=watershed_line)
        st.write("Watershed complete")
        tifffile.imwrite(save_path, watershed_test)
        end_time = time.time()
        st.write("Segmented image written")
        st.write("Segmentation Time Taken: {}".format(end_time-start_time))

    st.write(' ')
    st.write(' ')
    inf_path = st.text_input("Enter the file path to your inference .tif file.")
    st.write(' ')
    thresh = st.slider("Choose a threshold value to binarize your inference. (We suggest a high threshold value i.e. 0.75+).", min_value=0.00, max_value=1.00, step=0.01)
    st.write(' ')
    save_path = st.text_input("Enter the save path to your segmentation .tif file.")
    st.write(' ')
    st.write("A watershed line is a one-pixel wide line that separates the regions obtained by the watershed segmentation algorithm. The line has the label 0.")
    st.write(' ')
    watershedline = st.selectbox("Would you like to include a watershed line in your segmentation?", ["Yes", "No"])
    st.write(' ')
    if watershedline=="Yes":
        watershedLine = True
    else:
        watershedLine = False
    
    run = st.button("Run Watershed Segmentation")
    if run == True:
        with st.spinner("Segmentation Running"):
            watershed_seg(inf_path, thresh, save_path, watershed_line=watershedLine)
        st.balloons()
        st.markdown('#### Segmentation Complete!')
