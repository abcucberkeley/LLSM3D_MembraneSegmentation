import tifffile 
import numpy as np
import skimage 
from skimage import morphology
from tqdm import tqdm

z = input('Enter the path to your file: \n')
mask = tifffile.imread('{}'.format(z))

# Deleting values of 0 and 1 from unique values array for masks. 
arr = np.unique(mask)
arr = np.delete(arr, 0) # use as many times as needed 

def binarize(mask, arr): # arr is an array with all unique values of mask except 0 and 1
    mask_bw = 0 * mask # creating new array of the same size as mask, except all 0s
    for i in tqdm(range(len(arr))): 
        bw_i = (mask == arr[i]) # setting bw_i equal to every instance of a unique pixel value in mask
        bw_i = skimage.morphology.erosion(bw_i) # performing erosion
        mask_bw = mask_bw + bw_i 
    return mask_bw

binarized_mask = binarize(mask, arr)



# Save File to Desired Directory 
i = input('Enter your desired path for your file:')
tifffile.imwrite('{}'.format(i), binarized_mask)

# Or you can use decide to use a specific file path before running the program
# tifffile.imwrite('file_path', binarized_mask)


