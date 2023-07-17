import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Getting average pixel spacing
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

avgSpacing = [np.mean(xSpacing), np.mean(ySpacing)]
print("Average Spacing:", avgSpacing)
print("Average Slice Thickness:", np.mean(sliceThickness))

plt.hist(xSpacing)
plt.axvline(avgSpacing[0], color = "red", ls="--")
plt.title("Distribution of Pixel Spacing")
plt.xlabel("Pixel Spacing (mm)")
plt.show()

# plt.hist(ySpacing)
# plt.axvline(avgSpacing[1], color = "red", ls="--")
# plt.title("Distribution of y-axis Spacing")
# plt.show()

plt.hist(sliceThickness)
plt.axvline(np.mean(sliceThickness), color = "red", ls="--")
plt.title("Distribution of Slice Thickness")
plt.xlabel("Slice Thickness (mm)")
plt.show()