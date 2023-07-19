"""
This script resamples the original and segmented DICOM images to a uniform slice thickness.

The new position of the representative slice is redetermined using the original and new slice
thickness and the original position of the representative slice. The represenative slice and
the superior and inferior slices are then saved as DICOM files.
"""

import os
import pandas as pd
import SimpleITK as sitk
import pydicom
import numpy as np

NEW_SLICE_THICKNESS = 3.

# Get path to data folder
basePath = "/home/abbasshaikh/BAP1Project/Data/"
diseaseLaterality = pd.read_excel(os.path.join(basePath, "BAP1 data curation.xlsx"), 
                                    sheet_name = "Disease laterality - Feng")

# Get input and output paths
imagesPath = os.path.join(basePath, "HIRO-cases proper names")
outFolder = "TextureAnalysisFinal"
os.makedirs(os.path.join(basePath, outFolder), exist_ok = True)

# Create resampler object
resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)

for iCase in range(len(diseaseLaterality)):

    print("Processing Case: " + str(iCase + 1) + "/" + str(len(diseaseLaterality)) + " - " + diseaseLaterality.Case[iCase] + "\r")

    if not pd.isnull(diseaseLaterality["Middle slice"][iCase]):

        casePath = os.path.join(basePath, outFolder, diseaseLaterality.Case[iCase] + "_" + diseaseLaterality.Laterality[iCase])

        # Get path to all original DICOMs
        originalPath = os.path.join(imagesPath, diseaseLaterality.Case[iCase])
        originalImageNames = sorted([os.path.join(originalPath, file) for file in next(os.walk(originalPath))[2]])

        # Get path to all thorax-segmented DICOMs
        segmentedPath = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "SegmentedThorax")
        segmentedImageNames = sorted([os.path.join(segmentedPath, file) for file in next(os.walk(segmentedPath))[2]])

        # Get representative slice index
        repSlicePath = os.path.join(originalPath, diseaseLaterality["Middle slice"][iCase]) 
        sliceIndex = originalImageNames.index(repSlicePath)

        # Check ew have same number of original and segmented images
        assert len(originalImageNames) == len(segmentedImageNames)

        for imagePaths in [originalImageNames, segmentedImageNames]:
            
            # Read in one image for metadata
            dicom = pydicom.dcmread(imagePaths[0])
            
            # Create volume from 2D Slices
            size = (len(imagePaths), dicom.Rows, dicom.Columns)
            volumeArray = np.empty(size)
            
            for i in range(len(imagePaths)):
                volumeArray[i, :, :] = pydicom.dcmread(imagePaths[i]).pixel_array
                
            # Convert volume to Simple ITK
            volume = sitk.GetImageFromArray(volumeArray)
            volume.SetSpacing((dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness))

            # Specify resample parameters and resample volumes
            resampler.SetOutputSpacing((dicom.PixelSpacing[0], dicom.PixelSpacing[1], NEW_SLICE_THICKNESS))
            newSize = [round(size * (oldSpacing / newSpacing)) for 
                    size, newSpacing, oldSpacing in zip(volume.GetSize(), resampler.GetOutputSpacing(), volume.GetSpacing())]
            resampler.SetSize(newSize)
            resampledVolume = resampler.Execute(volume)

            # Get new slice index based on new size
            newSliceIndex = round((sliceIndex + 1) / volume.GetSize()[2] * resampler.GetSize()[2]) - 1

            # Infer new slice location
            newSliceLocation = dicom.SliceLocation + NEW_SLICE_THICKNESS * newSliceIndex 

            # Get paths to save new slices
            if imagePaths[0][-11:-8] == "Img":
                outPath = os.path.join(casePath, "OriginalImgs_Resampled")
                os.makedirs(outPath, exist_ok = True)

                repName = "Img001_" + str(((newSliceIndex + 1))).zfill(4)
                supName = "Img001_" + str(((newSliceIndex + 1) - 1)).zfill(4)
                infName = "Img001_" + str(((newSliceIndex + 1) + 1)).zfill(4)
                
            elif imagePaths[0][-11:-8] == "Thx":
                outPath = os.path.join(casePath, "SegmentedThorax_Resampled")
                os.makedirs(outPath, exist_ok = True)

                repName = "Thx001_" + str(((newSliceIndex + 1))).zfill(4)
                supName = "Thx001_" + str(((newSliceIndex + 1) - 1)).zfill(4)
                infName = "Thx001_" + str(((newSliceIndex + 1) + 1)).zfill(4)
                
            #  Get representative and neighboring slices as numpy arrays
            repSlice = (sitk.GetArrayFromImage(resampledVolume[:, :, newSliceIndex])).astype(np.int16)
            supSlice = (sitk.GetArrayFromImage(resampledVolume[:, :, newSliceIndex - 1])).astype(np.int16)
            infSlice = (sitk.GetArrayFromImage(resampledVolume[:, :, newSliceIndex + 1])).astype(np.int16)
            

            # Modify DICOM header information with updated slice thickness 
            # and location for each slice and save to a new DICOM file.
            dicom.SliceThickness = NEW_SLICE_THICKNESS

            dicom.SliceLocation = newSliceLocation
            dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -newSliceLocation]
            dicom.PixelData = repSlice.tobytes()
            dicom.save_as(os.path.join(outPath, repName))

            dicom.SliceLocation = newSliceLocation - NEW_SLICE_THICKNESS
            dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -(newSliceLocation - NEW_SLICE_THICKNESS)]
            dicom.PixelData = supSlice.tobytes()
            dicom.save_as(os.path.join(outPath, supName))

            dicom.SliceLocation = newSliceLocation + NEW_SLICE_THICKNESS
            dicom.ImagePositionPatient = [dicom.ImagePositionPatient[0], dicom.ImagePositionPatient[1], -(newSliceLocation + NEW_SLICE_THICKNESS)]
            dicom.PixelData = infSlice.tobytes()
            dicom.save_as(os.path.join(outPath, infName))