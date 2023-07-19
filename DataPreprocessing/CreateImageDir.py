"""
This script creates a directory of images from the raw data that is configured for
mesothelioma segmentation and texture feature extraction.

The original, thorax-segmented, and preprocessed images are copied to the new directory.
"""

import os
import shutil
import pandas as pd

# Get path to data folder
basePath = "/home/abbasshaikh/BAP1Project/Data/"
diseaseLaterality = pd.read_excel(os.path.join(basePath, "BAP1 data curation.xlsx"), 
                                    sheet_name = "Disease laterality - Feng")

# Get input and output paths
imagesPath = os.path.join(basePath, "HIRO-cases proper names")
outFolder = "TextureAnalysisFinal"
os.makedirs(os.path.join(basePath, outFolder), exist_ok = True)

for iCase in range(len(diseaseLaterality)):
    
    print("Processing Case: " + str(iCase + 1) + "/" + str(len(diseaseLaterality)) + " - " + diseaseLaterality["Case"][iCase] + "\r")

    if not pd.isnull(diseaseLaterality["Middle slice"][iCase]):

        casePath = os.path.join(basePath, outFolder, diseaseLaterality.Case[iCase] + "_" + diseaseLaterality.Laterality[iCase])
        os.makedirs(casePath, exist_ok = True)

        # Get paths to original images
        repName = diseaseLaterality["Middle slice"][iCase]
        sliceNum = int(repName[-4:])

        supName = repName[:-4] + str((sliceNum - 1)).zfill(4)
        infName = repName[:-4] + str((sliceNum + 1)).zfill(4)

        repImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], repName)
        supImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], supName)
        infImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], infName)

        # Copy original images
        outPath = os.path.join(casePath, "OriginalImgs")
        os.mkdir(outPath)

        shutil.copy(repImage, outPath)
        shutil.copy(supImage, outPath)
        shutil.copy(infImage, outPath)

        # Get paths to segmented thorax images
        repNameThx = "Thx" + repName[3:]
        supNameThx = "Thx" + supName[3:]
        infNameThx = "Thx" + infName[3:]

        repImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "SegmentedThorax", repNameThx)
        supImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "SegmentedThorax", supNameThx)
        infImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "SegmentedThorax", infNameThx)

        # Copy segmented thorax images
        outPath = os.path.join(casePath, "SegmentedThorax")
        os.mkdir(outPath)

        shutil.copy(repImage, outPath)
        shutil.copy(supImage, outPath)
        shutil.copy(infImage, outPath)

        # Get paths to preprocessed images
        repImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", repNameThx + ".tif")
        supImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", supNameThx + ".tif")
        infImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", infNameThx + ".tif")

        # Copy preprocessed images
        outPath = os.path.join(casePath, "PreprocessedImgs")
        os.mkdir(outPath)

        shutil.copy(repImage, outPath)
        shutil.copy(supImage, outPath)
        shutil.copy(infImage, outPath)


    
