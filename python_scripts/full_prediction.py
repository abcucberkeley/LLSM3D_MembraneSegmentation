from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import tifffile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import seaborn as sns
import math


# Loading Model
i = input('Please input the file path to your 3D U-Net model \n')
model = tf.keras.models.load_model('{}'.format(i))

# Inputting Raw (Scaled) Image

k = input('Please input the file path to your raw (scaled) image \n')
img = tifffile.imread('{}'.format(k))

def pd_preprocessing(pd):
    label = pd
    label = np.squeeze(label)
    label = np.transpose(label, (3, 0, 1, 2))
    label = label[1]
    return label

def full_volume_prediction(img, imgh, imgw, imgl, chunkh, chunkl, chunkw, OL):

    if OL%2 !=0:
        print('ERROR: Please enter an even OL value for the program to run.')
    else:
        a = imgh/chunkh
        a = math.ceil(a)
        b = imgl/chunkl
        b = math.ceil(b)
        c = imgw/chunkw
        c = math.ceil(c)
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
                    else:
                        xmax[num_chunks] = imgw

                    ymin[num_chunks] = (((y)*(chunkl-OL)))
                    if y < chunkl: 
                        ymax[num_chunks] = (chunkl*(y+1)-(OL*(y)))
                    else:
                        ymax[num_chunks] = imgl

                    zmin[num_chunks] = (((z)*(chunkh-OL)))
                    if z < chunkh: 
                        zmax[num_chunks] = (chunkh*(z+1)-(OL*(z)))
                    else:
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
            if (cropped_img.shape ==(chunkh, chunkl, chunkw)):
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
imgw = int(input('Please input the full image array y dimension size \n'))
imgl = int(input('Please input the full image array x dimension size \n'))
chunkh = int(input('Please input the image subvolume z dimension size \n'))
chunkl = int(input('Please input the image subvolume y dimension size \n'))
chunkw = int(input('Please input the image subvolume x dimension size \n'))
OL = int(input('Please input your preffered number of overlap pixels \n'))
save_path = input('Please enter your desired file path for the final full prediction \n')
full_pred = full_volume_prediction(img, imgh, imgw, imgl, chunkh, chunkw, chunkl, OL)
tifffile.imwrite('{}'.format(save_path), full_pred)


