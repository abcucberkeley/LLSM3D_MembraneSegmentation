"""

LLSM 3D Membrane Segmentation using 3D U-Net
Author: Zeeshan Patel
Instructions: Complete all input fields when you run the script. You may run the script in multiple 
              compute nodes if you have several images to create labels for. This script takes 
              approximately 5 minutes to generate a label for one bioimage, on average.

"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import tifffile
import math
from tensorflow.keras import backend as K


# U-Net Model Loss Function

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

# Loading Model
i = input('Please input the file path to your 3D U-Net model \n')
model = tf.keras.models.load_model('{}'.format(i), custom_objects={'bce_dice_loss': bce_dice_loss})

# Inputting Raw (Scaled) Image

k = input('Please input the file path to your raw (scaled) image \n')
img = tifffile.imread('{}'.format(k))

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
    
imgh = int(input('Please input the full image array z dimension size \n'))
imgl = int(input('Please input the full image array y dimension size \n'))
imgw = int(input('Please input the full image array x dimension size \n'))
chunkh = int(input('Please input the image subvolume z dimension size \n'))
chunkl = int(input('Please input the image subvolume y dimension size \n'))
chunkw = int(input('Please input the image subvolume x dimension size \n'))
OL = int(input('Please input your preffered number of overlap pixels \n'))
save_path = input('Please enter your desired file path for the final full prediction \n')
full_pred = full_volume_prediction(img, imgh, imgl, imgw, chunkh, chunkw, chunkl, OL)
tifffile.imwrite('{}'.format(save_path), full_pred)
