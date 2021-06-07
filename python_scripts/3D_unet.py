"""

LLSM 3D Membrane Segmentation using 3D U-Net
Author: Zeeshan Patel
Instructions: Before running this script, please configure your training image filepaths (lines 53 and 55) to match the example format.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
import sys
import numpy as np 
import tifffile
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import math
import datetime

begin_time = datetime.datetime.now()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Pre-defined parameters 
seed = 42
np.random.seed = seed

IMG_WIDTH = 112
IMG_HEIGHT = 112
IMG_DEPTH = 32
IMG_CHANNELS = 1
NUM_IMGS = 2508
TEST_SIZE = 0.2
BATCH_SIZE = 8
NUM_EPOCHS = 250
model_save_location = input("Enter the file path for the model h5 file: ") # Ex: /my/machine/.../model.h5

#####################################################################################################################
#####################################################################################################################

# Loading Training Data
x = []
y = []
count = 0
while count < NUM_IMGS:
    
    img = tifffile.imread('/some/path/cropped_imgs/img{}.tif'.format(count)) # Make sure to configure your own path before running the script
    x.append(img)
    mask = tifffile.imread('/some/path/cropped_masks/mask{}.tif'.format(count)) # Make sure to configure your own path before running the script
    y.append(mask)
    
    count+=1

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=TEST_SIZE)
x_train = np.array(x_train, dtype=np.uint8)
y_train = np.array(y_train, dtype=np.bool)
x_test = np.array(x_test, dtype=np.uint8)
y_test = np.array(y_test, dtype=np.bool)

print('Data Loaded successfully!')

#####################################################################################################################
#####################################################################################################################

# 3D U-Net Model Functions

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

#####################################################################################################################
#####################################################################################################################

# Distributed 3D U-Net Architecture 

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

# You can uncomment the below lines if you are interested in using TensorBoard for more advanced analytics on your model.
# If you want to use a callback, make sure to add the callback parameter in model.fit()

# log_dir = 'logs/fit'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
u_net = model.fit(train_dataset, epochs=NUM_EPOCHS, verbose=1, validation_data=test_dataset)

model.save(model_save_location)

#####################################################################################################################
#####################################################################################################################

# Training/Validation Accuracy + Loss Plot

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
plt.savefig('unet_test_distributed.png', format='png', dpi=500)
plt.show() 

print("Time Taken for Model Training: ", datetime.datetime.now() - begin_time)

