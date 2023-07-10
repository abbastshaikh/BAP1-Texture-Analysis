import os
import pandas as pd
import SimpleITK as sitk
import pydicom
import numpy as np

NEW_SLICE_THICKNESS = 3.

# Set working director to data folder
basePath = "/home/abbasshaikh/BAP1Project/Data/"
diseaseLaterality = pd.read_excel(os.path.join(basePath, "BAP1 data curation.xlsx"), 
                                    sheet_name = "Disease laterality - Feng")

# Get input and output paths
imagesPath = os.path.join(basePath, "HIRO-cases proper names")
outFolder = "TextureAnalysis"
os.makedirs(os.path.join(basePath, outFolder), exist_ok = True)

# Create resampler object
resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)

for iCase in range(len(diseaseLaterality)):

    print("Processing Case: " + str(iCase + 1) + "/" + str(len(diseaseLaterality)) + " - " + diseaseLaterality.Case[iCase] + "\r")

    if not pd.isnull(diseaseLaterality["Middle slice"][iCase]):

        casePath = os.path.join(basePath, outFolder, diseaseLaterality.Case[iCase] + "_" + diseaseLaterality.Laterality[iCase])

        # Get path to all thorax-segmented DICOMs
        caseImagePath = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "SegmentedThorax")
        imageNames = sorted([os.path.join(caseImagePath, file) for file in next(os.walk(caseImagePath))[2]])
        
        # Get path to representative slice and slice 
        sliceName = "Thx" + diseaseLaterality["Middle slice"][iCase][3:]
        repSlicePath = os.path.join(caseImagePath, sliceName) 
        sliceIndex = imageNames.index(repSlicePath)

        # Read in one image for metadata
        dicom = pydicom.dcmread(imageNames[0])

        # Create volume from 2D Slices
        size = (len(imageNames), dicom.Columns, dicom.Rows)
        volumeArray = np.empty(size)

        for i in range(len(imageNames)):
            image = sitk.ReadImage(imageNames[i])
            volumeArray[i, :, :] = sitk.GetArrayFromImage(image)

        # Convert volume to Simple ITK
        volume = sitk.GetImageFromArray(volumeArray)
        volume.SetSpacing((dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness))

        # Specify resample parameters
        resampler.SetOutputSpacing((dicom.PixelSpacing[0], dicom.PixelSpacing[1], NEW_SLICE_THICKNESS))
        newSize = [round(size * (oldSpacing / newSpacing)) for 
                size, newSpacing, oldSpacing in zip(volume.GetSize(), resampler.GetOutputSpacing(), volume.GetSpacing())]
        resampler.SetSize(newSize)

        # Get new slice index based on new size
        newSliceIndex = round((sliceIndex + 1) / volume.GetSize()[2] * resampler.GetSize()[2]) - 1

        # Resample image
        resampledVolume = resampler.Execute(volume)

        newRepSlice = resampledVolume[:, :, newSliceIndex]
        newSupSlice = resampledVolume[:, :, newSliceIndex - 1]
        newInfSlice = resampledVolume[:, :, newSliceIndex + 1]

        # Infer new slice location
        newSliceLocation = dicom.SliceLocation + NEW_SLICE_THICKNESS * newSliceIndex
        
        # Save slices
        outPath = os.path.join(casePath, "ResampledImgs_" + str(int(NEW_SLICE_THICKNESS)) + "mm")
        os.makedirs(outPath, exist_ok = True)

        repName = "Thx001_" + str(((newSliceIndex + 1))).zfill(4)
        supName = "Thx001_" + str(((newSliceIndex + 1) - 1)).zfill(4)
        infName = "Thx001_" + str(((newSliceIndex + 1) + 1)).zfill(4)

        dicom.SliceThickness = NEW_SLICE_THICKNESS

        dicom.SliceLocation = newSliceLocation
        dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -newSliceLocation]
        dicom.PixelData = sitk.GetArrayFromImage(newRepSlice).astype(np.int16)
        dicom.save_as(os.path.join(outPath, repName))

        dicom.SliceLocation = newSliceLocation - NEW_SLICE_THICKNESS
        dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -(newSliceLocation - NEW_SLICE_THICKNESS)]
        dicom.PixelData = sitk.GetArrayFromImage(newSupSlice).astype(np.int16)
        dicom.save_as(os.path.join(outPath, supName))

        dicom.SliceLocation = newSliceLocation + NEW_SLICE_THICKNESS
        dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -(newSliceLocation + NEW_SLICE_THICKNESS)]
        dicom.PixelData = sitk.GetArrayFromImage(newInfSlice).astype(np.int16)
        dicom.save_as(os.path.join(outPath, infName))