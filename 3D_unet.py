from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np 
import tifffile
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

seed = 42
np.random.seed = seed


# Loading Training and Testing Data

# Best size for training dataset
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 64
IMG_CHANNELS = 1

train_num = int(input('How many training images will you use? \n'))
test_num = int(input('How many testing images will you use? \n'))

x_train = np.zeros((train_num, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
y_train = np.zeros((train_num, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

x_test = np.zeros((test_num, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
y_test = np.zeros((test_num, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)


# Using 1800 images from training dataset in cluster -- data pipeline needs to be modified 
counter = 0
while counter < 600: 
    x_train[counter] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_imgs/raw_data/no_noise/cropped_img_'+f"{counter:03}" + '.tif')
    y_train[counter] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_masks/raw_data/no_noise/cropped_mask_'+f"{counter:03}" + '.tif')
    counter +=1
counter = 0
while counter < 600: 
    x_train[counter+600] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_imgs/raw_data/noise_level_1/cropped_img_'+f"{counter:03}" + '.tif')
    y_train[counter+600] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_masks/raw_data/noise_level_1/cropped_mask_'+f"{counter:03}" + '.tif')
    counter +=1
counter = 0
while counter < 600: 
    x_train[counter+1200] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_imgs/raw_data/noise_level_2/cropped_img_'+f"{counter:03}" + '.tif')
    y_train[counter+1200] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_masks/raw_data/noise_level_2/cropped_mask_'+f"{counter:03}" + '.tif')
    counter +=1
counter = 0
while counter < 25:
    x_test[counter] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_imgs/raw_data/no_noise/cropped_img_' + f"{counter:03}" + '.tif')
    y_test[counter] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_masks/raw_data/no_noise/cropped_mask_' + f"{counter:03}" + '.tif')
    counter+=1
counter = 0
while counter < 25:
    x_test[counter+25] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_imgs/raw_data/noise_level_1/cropped_img_' + f"{counter:03}" + '.tif')
    y_test[counter+25] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_masks/raw_data/noise_level_1/cropped_mask_' + f"{counter:03}" + '.tif')
    counter+=1
counter = 0
while counter < 25:
    x_test[counter+50] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_imgs/raw_data/noise_level_2/cropped_img_' + f"{counter:03}" + '.tif')
    y_test[counter+50] = tifffile.imread('/clusterfs/fiona/zeeshan/cropped_test_masks/raw_data/noise_level_2/cropped_mask_' + f"{counter:03}" + '.tif')
    counter+=1
print('Done!')

###########################################################################################################################
###########################################################################################################################

# Building 3D U-Net


# Contraction Path
inputs = tf.keras.layers.Input((IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
c1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
c1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

c2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
c2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)

c3 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
c3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3)

c4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)
c4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)

# Expansive Path
u6 = tf.keras.layers.Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same')(c4)
u6 = tf.keras.layers.concatenate([u6, c3])
c6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.BatchNormalization(axis=-1)(c6)
c6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
c6 = tf.keras.layers.BatchNormalization(axis=-1)(c6)

u7 = tf.keras.layers.Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c2])
c7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.BatchNormalization(axis=-1)(c7)
c7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
c7 = tf.keras.layers.BatchNormalization(axis=-1)(c7)

u8 = tf.keras.layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c1])
c8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.BatchNormalization(axis=-1)(c8)
c8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
c8 = tf.keras.layers.BatchNormalization(axis=-1)(c8)

outputs = tf.keras.layers.Conv3D(3, (1, 1, 1), activation='sigmoid')(c8)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

###########################################################################################################################
###########################################################################################################################

# Callbacks

log_dir = 'logs/fit'
i = input('Choose a filepath for your Model Checkpoint \n')
filepath = '{}'.format(i)
# List of Callbacks -- Add more Callbacks if necessary
callback_list = [
tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor = 'val_accuracy', 
                                   save_best_only = True)
]

# If you would like to only use the tensorboard_callback, uncomment the following line and comment out the callback_list
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


###########################################################################################################################
###########################################################################################################################

# Model Training


# Change hyperparameters as needed -- these hyperparameters worked best for 3D live-cell membrane data acquired from AO-LLSM

j = input('Choose a filepath to save your final model \n')
u_net = model.fit(x_train, y_train, batch_size=1, epochs=150, verbose=1, validation_split=0.15, steps_per_epoch=15)

model.save('{}'.format(j))

###########################################################################################################################
###########################################################################################################################

# Plotting Model Performance

# Change figsize and x/y_ticks as needed

k = input('Choose a filepath to save a PNG version of your plot \n')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 25))
t = f.suptitle('3D U-Net Performance on Cell Membrane Data', fontsize=24)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(u_net.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, u_net.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, u_net.history['val_accuracy'], label='Validation Accuracy')
# ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value', fontsize=15)
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_title('Accuracy', fontsize=16)
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, u_net.history['loss'], label='Train Loss')
ax2.plot(epoch_list, u_net.history['val_loss'], label='Validation Loss')
# ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value', fontsize=15)
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_title('Loss', fontsize=16)
l2 = ax2.legend(loc="best")
plt.savefig('{}'.format(k), format='png')
plt.show() 


