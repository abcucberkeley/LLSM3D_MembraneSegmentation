import tifffile 
import numpy as np
import skimage 
from skimage import morphology
from tqdm import tqdm

# Load Raw Mask
z = input('Enter the path to the raw, original mask: \n')
mask = tifffile.imread('{}'.format(z))

# Save Eroded Binary Mask to Desired Directory 
i = input('Enter your desired path for the Eroded Binary Mask: \n')

# Save Final Binary Mask to Desired Directory 
l = input('Enter your desired path for the Final Binary Mask: \n')

# Removing values of 0 and 1 from unique values array for masks. 
arr = np.unique(mask)
arr = arr[arr != 0]
arr = arr[arr != 1]

# Binarizing Mask using Erosion
def binarize(mask, arr): # arr is an array with all unique values of mask except 0 and 1
    mask_bw = 0 * mask # creating new array of the same size as mask, except all 0s
    for i in tqdm(range(len(arr))): 
        bw_i = (mask == arr[i]) # setting bw_i equal to every instance of a unique pixel value in mask
        bw_i = skimage.morphology.erosion(bw_i) # performing erosion
        mask_bw = mask_bw + bw_i 
    return mask_bw

# Removing Cytosolic Values from Binary Mask
def rm_cytosol(mask, binary_mask):
    b_mask = mask>1 # mask --> original mask
    final_mask = binary_mask - b_mask
    return final_mask

def conv_to_boolean(final_mask):  # Converting final mask to boolean 
    final_mask.dtype = np.bool 
    final_mask = final_mask[::, ::, 1::2]
    return final_mask


# Eroded Binary Mask (With Cytosolic Values)
binarized_mask = binarize(mask, arr)

tifffile.imwrite('{}'.format(i), binarized_mask)


# Binary Mask (Without Cytosolic Values)
final_mask = rm_cytosol(mask, binary_mask)
final_mask = conv_to_boolean(final_mask)

tifffile.imwrite('{}'.format(l), final_mask)
