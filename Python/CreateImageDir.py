import os
import shutil
import pandas as pd

# Set working director to data folder
basePath = "/home/abbasshaikh/BAP1Project/Data/"
diseaseLaterality = pd.read_excel(os.path.join(basePath, "BAP1 data curation.xlsx"), 
                                    sheet_name = "Disease laterality - Feng")

# Get input and output paths
imagesPath = os.path.join(basePath, "HIRO-cases proper names")
outFolder = "TextureAnalysis"
os.mkdir(os.path.join(basePath, outFolder))

for iCase in range(len(diseaseLaterality)):

    print("Processing Case: " + str(iCase + 1) + "/" + str(len(diseaseLaterality)) + "\r")

    if not pd.isnull(diseaseLaterality["Middle slice"][iCase]):

        casePath = os.path.join(basePath, outFolder, diseaseLaterality.Case[iCase] + "_" + diseaseLaterality.Laterality[iCase])
        os.mkdir(casePath)

        # Save original images
        imName = diseaseLaterality["Middle slice"][iCase]
        sliceNum = int(imName[-4:])

        supName = imName[:-4] + str((sliceNum - 1)).zfill(4)
        infName = imName[:-4] + str((sliceNum + 1)).zfill(4)

        image = os.path.join(imagesPath, diseaseLaterality.Case[iCase], imName)
        supImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], supName)
        infImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], infName)

        outPath = os.path.join(casePath, "OriginalImgs")
        os.mkdir(outPath)

        shutil.copy(image, outPath)
        shutil.copy(supImage, outPath)
        shutil.copy(infImage, outPath)

        # Save Preprocessed images
        imName = "Thx" + diseaseLaterality["Middle slice"][iCase][3:]
        sliceNum = int(imName[-4:])

        supName = imName[:-4] + str((sliceNum - 1)).zfill(4)
        infName = imName[:-4] + str((sliceNum + 1)).zfill(4)

        image = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", imName + ".tif")
        supImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", supName + ".tif")
        infImage = os.path.join(imagesPath, diseaseLaterality.Case[iCase], "PreprocessedImgs", infName + ".tif")

        outPath = os.path.join(casePath, "PreprocessedImgs")
        os.mkdir(outPath)

        shutil.copy(image, outPath)
        shutil.copy(supImage, outPath)
        shutil.copy(infImage, outPath)


    
