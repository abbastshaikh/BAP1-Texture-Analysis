import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Getting average pixel spacing and slice thickness for dataset
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