"""
This script includes miscellaneous exploratory plots about the dataset, 
including about the distribution of pixel spacing, slice thickness, 
and mutation and IHC status, and the effects of preprocessing filters 
(LoG, wavelet, LBP) on images.

Written by Abbas Shaikh, Summer 2023
"""

import os
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import radiomics

### Pixel Spacing Statistics ###
dataPath = r"D:\BAP1\Data\TextureAnalysisFinal"
reader = sitk.ImageFileReader()
spacing = []
sliceThickness = []

for case in os.listdir(dataPath):
    
    imgFolderPath = os.path.join(dataPath, case, "SegmentedThorax")
    imgPath = os.path.join(imgFolderPath, os.listdir(imgFolderPath)[0])
    
    reader.SetFileName(imgPath)
    image = reader.Execute()[:, :, 0]
    
    spacing.append(image.GetSpacing())
    
    dicom = pydicom.dcmread(imgPath)
    sliceThickness.append(dicom.SliceThickness)
    
xSpacing = [s[0] for s in spacing]  
ySpacing = [s[1] for s in spacing]   

print("Average Spacing:", [np.mean(xSpacing), np.mean(ySpacing)])
print("Average Slice Thickness:", np.mean(sliceThickness))

plt.hist(xSpacing)
plt.axvline(np.mean(xSpacing), color = "red", ls="--")
plt.title("Distribution of Pixel Spacing (x-axis)")
plt.xlabel("Pixel Spacing (mm)")
plt.show()

plt.hist(ySpacing)
plt.axvline(np.mean(ySpacing), color = "red", ls="--")
plt.title("Distribution of Pixel Spacing (y-axis)")
plt.xlabel("Pixel Spacing (mm)")
plt.show()

plt.hist(sliceThickness)
plt.axvline(np.mean(sliceThickness), color = "red", ls="--")
plt.title("Distribution of Slice Thickness")
plt.xlabel("Slice Thickness (mm)")
plt.show()


### Mutation Status and IHC Status Statistics ###
dataPath = r"D:\BAP1\Data"
labels = pd.read_csv(os.path.join(dataPath, "BAP1Labels.csv"))

labels["IHC BAP1 Status"].loc[labels["IHC BAP1 Status"] == "Lost"] = "Loss"

table = pd.crosstab(labels["Somatic BAP1 Mutation Status"], labels["IHC BAP1 Status"])

fig, ax = plt.subplots(figsize = (8, 8))
plt.title("BAP1 IHC and Somatic Mutation Status")

bar = table.plot.bar(stacked = True, ax = ax,
               xlabel = "Somatic Mutation Status",
               ylabel = "Count")

plt.legend(["Loss", "Not Done", "Retained"], title = "IHC Status")

plt.show()


### Exploring effects of preprocessing filters ###
image = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\SegmentedThorax_Resampled\Thx001_0056")[:, :, 0]
mask = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\Masks_Resampled\imgs_2_18000101_CT_Axial_Chest_Recon_right_corrected_contour_1.tif")

# Adjusting background pixels for visualization
imageArray = sitk.GetArrayFromImage(image)
imageArray[imageArray < -1000] = -1000
image = sitk.GetImageFromArray(imageArray)

# Show original image and tumor segmentation
plt.imshow(sitk.GetArrayFromImage(image), cmap = "gray")
plt.title("Original Image")
plt.show()

plt.imshow(sitk.GetArrayFromImage(mask), cmap = "gray")
plt.title("Tumor Mask")
plt.show()

# Plotting images with Laplacian of Gaussian (LoG) filters
for sigma in [0.5, 1, 2, 4, 8]:
    LoGImage = sitk.LaplacianRecursiveGaussian(image, sigma = sigma)
    
    plt.imshow(sitk.GetArrayFromImage(LoGImage), cmap = "gray")
    plt.title("Laplacian of Gaussian, Sigma = " + str(sigma))
    plt.show()

# Plotting images with Local Binary Pattern (LBP) filters
for radius in [1, 2, 4]:
    LBPImage = next(radiomics.imageoperations.getLBP2DImage(image, mask, lbp2DRadius = radius))[0]
    plt.imshow(sitk.GetArrayFromImage(LBPImage), cmap = "gray")
    plt.title("Local Binary Pattern, Radius = " + str(radius))
    plt.show()

# Plotting images with wavelet filters
for wavelet in ['haar', 'dmey', 'coif1', 'sym2', 'db2', 'bior1.1', 'rbio1.1']:
    
    waveletTransform = radiomics.imageoperations.getWaveletImage(image, mask, wavelet = wavelet)
    
    # Iterating through all decompositions
    while True:
        try:
            waveletImage = next(waveletTransform)
            plt.imshow(sitk.GetArrayFromImage(waveletImage[0]), cmap = "gray")
            plt.title(wavelet + " " + waveletImage[1])
            plt.show()
        except StopIteration:
            break