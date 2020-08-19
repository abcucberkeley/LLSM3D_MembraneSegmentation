import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy 
import math 
from tifffile import imwrite



# Load Raw Image
i = input('Enter the path to your raw image: \n')
raw_image = tifffile.imread('{}'.format(i))

# File path to save 8-bit Image
k = input('Enter your desired file path for the new scaled image: \n')

# Identifying Percentiles for Intensity

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
for i in tqdm(range(len(ranges))): 
    intensity_at_percentile = np.percentile(raw_image, ranges[i])
    intensity[i] = intensity_at_percentile

# Code to View Intensity Profile (Not Necessary)
# plt.figure(figsize=(10,10))
# plt.title('Intensity vs Percentile', fontsize=26)
# plt.xlabel('Intensity', fontsize=20)
# plt.ylabel('Percentile', fontsize=20)
# plt.yscale('log')
# plt.plot(intensity, ranges)
# plt.show()

# Normalization Function 
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

# Normalizing Raw Image
i = intensity[153] # Using 99.95th percentile pixel intensity value for RangeIn
img_scaled = scaleContrast(raw_image, rangeIn=(0, i), rangeOut=(0,255))

tifffile.imwrite('{}'.format(k), img_scaled)

