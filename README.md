# AO-LLSM 3D Membrane Detection and Segmentation

## Abstract

Adaptive Optical Lattice light-sheet microscopy (AO-LLSM) is a 3D live-cell imaging method that reduces damage to the physiological state
of the biological specimen due to phototoxicity while also maintaining high-resolution image quality by avoiding issues with aberrations (Gao et al., 2019;
Liu et al., 2018). AO-LLSM delivers images in high spatiotemporal resolution, which allows for detailed analysis of individual cells in complex, multicellular
organisms. However, identifying cell boundaries in dense, multicellular organisms can be a tedious task. In order to understand cellular processes, it is vital
to properly identify and segment individual cell membranes. Precise labeling will enable the isolation of any cell in images of dense, multicellular organisms,
allowing researchers to analyze their cellular dynamics, cell morphologies, and organelle processes. We outline a 3D-live cell membrane segmentation
method using deep learning with 3D U-Net. We developed our own image normalization and ground truth label binarization algorithms for data
preprocessing using core frameworks such as NumPy and scikit-image. To generate training and testing datasets, we created random augmentation
algorithms that utilized Gaussian noise functions and TensorFlow based image augmentation functions. To develop the 3D U-Net, we utilized the
Tensorflow-based Keras API and the original neural network architecture (Özgün Çiçek et al., 2016). To use our tool on large, volumetric datasets, we
developed a prediction script that can take in 3D cell membrane images of any size and produce a predicted label for the image. **Our 3D U-Net was able
to generate segmentation labels in less than five minutes, with accuracies around 98.10%. The overall process allows for fast image processing and rapid
neural network training, providing both a time and cost-efficient, automated method for 3D cell membrane detection and segmentation.** 


## Code
This code can be modified as needed for user-specific needs. Most users will only need to modify data pipelines and file paths. We have multiple Python scripts that are used in the following order:

### Data Preprocessing:

  1. Mask Binarization (mask_conversion.py)
  2. Image Normalization (img_normalization.py)
  3. Data Augmentation + Preparation (data_augmentation.py)

### Model Training:

  4. 3D U-Net (3D_unet.py)

### Model Testing + Predicting (Interchangeable):

  5. Interactive Testing on Training + Testing Datasets (model_interactive_pred.ipynb)
  6. Full Volume Prediction (full_prediction.py)
  
## Streamlit Web Application

We have also developed a web application that can be used to run all of the Python scripts mentioned above. To run this application, clone the repository and run the following commands:

```
conda create llsm-streamlit
conda activate llsm-streamlit
cd python_scripts
conda install --file requirements.txt
streamlit run app.py
```

## Additional Documentation

To learn more about the specific details of each script and all of the parameters, please refer to our <a href="https://docs.google.com/document/d/1pYugyeuD_Ypg7Xc-aFosiLLn-595bRr3oNI_ojKiK1A/edit?usp=sharing">docs</a>.

## Project Results

Please refer to the this <a href="https://github.com/abcucberkeley/LLSM3D_MembraneSegmentation/blob/master/poster.pdf" download>project poster</a> for more information regarding our experimentation methods and results. If you have any inquiries regarding our project or code, feel free to contact Zeeshan Patel (zeeshanp [at] berkeley.edu.) 
  

## References 

(1) Özgün Çiçek et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. arXiv:1606.06650. (2) Gao et al. (2019). Cortical column and whole-brain imaging with molecular contrast and nanoscale resolution. *Science* Vol. 363, Issue 6424. (3) Liu et al. (2018). Observing the cell in its native state: Imaging subcellular dynamics in multicellular organisms. *Science* Vol. 360, Issue 6386. (4) Aguet et al. (2016) Membrane dynamics of dividing cells imaged by lattice light-sheet microscopy. *Molecular Biology of the Cell* 2016 27:22, 3418-3435. 
  
