import os
import pandas as pd
import radiomics
import SimpleITK as sitk
import numpy as np
from FeatureExtractionUtil import fullImageFeatureExtraction, preprocess
from scipy.stats import pearsonr

np.random.seed(0)
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

# Configurations
dataPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Data\TextureAnalysis"
outPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Experiments"
maskThreshold = 0.01
pixelSpacing = 0.75

# Create radiomics extractor object
radiomicsExtractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
for feature_class in ['firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']:
    radiomicsExtractor.enableFeatureClassByName(feature_class)
    
# Normalize to zero mean and unit variance (numpy)
def normalize (data):
    return (data - data.mean()) / data.std()

# Create dictionary to store texture features
nonErodedFeatures = {}
erodedFeatures = {}
    
# Iterate over cases (each dir = case)
cases = os.listdir(dataPath)
for caseIdx in range(len(cases)):
    
    caseName = cases[caseIdx].rsplit("_", 1)[0]
    print("Processing Case " + str(caseIdx + 1) + "/" + str(len(cases)) + ": ", caseName)
    
    originalImgsPath = os.listdir(os.path.join(dataPath, cases[caseIdx], "OriginalImgs"))
    probMapsPath = os.listdir(os.path.join(dataPath, cases[caseIdx], "prob_maps"))
    
    # List to store feature dictionaries for each slice
    sliceFeatures = []
    erodedSliceFeatures = []
    
    # Iterate over each slice per case
    for idx in range(len(originalImgsPath)):
        
        imgPath = os.path.join(dataPath, cases[caseIdx], "OriginalImgs", originalImgsPath[idx])
        probPath = os.path.join(dataPath, cases[caseIdx], "prob_maps", probMapsPath[idx])
        
        # Read in image
        image = sitk.ReadImage(imgPath)[:, :, 0]
        
        # Handling edge case, original image size doesn't match
        if caseName == "29_18000101_CT_AXL_W":
            image = image[:, -512:]
        
        # Read in probability map and threshold to produce mask
        probMap = sitk.ReadImage(probPath)
        mask = sitk.BinaryThreshold(probMap, lowerThreshold = maskThreshold)
        
        # Preprocess image and mask (normalizes pixel spacing)
        image, mask = preprocess(image, mask,  
                                 pixelSpacing,
                                 "", 0, False)
        
        # Get texture features
        featuresExtracted = fullImageFeatureExtraction(image, mask, True, True, radiomicsExtractor)
        
        # Get texture features for eroded mask
        erodeFilter = sitk.BinaryErodeImageFilter()
        erodeFilter.SetKernelRadius(1)
        erodedMask = erodeFilter.Execute(mask)
        erodedFeaturesExtracted = fullImageFeatureExtraction(image, erodedMask, True, True, radiomicsExtractor)
        
        if featuresExtracted:
            sliceFeatures.append(featuresExtracted)
        if erodedFeaturesExtracted:
            erodedSliceFeatures.append(erodedFeaturesExtracted)
            
    # Average features across all slices and add to overall features values dictionary
    if len(sliceFeatures) > 0:
        for feat in sliceFeatures[0].keys():
            avgFeatureValue = np.mean([float(featureDict[feat]) for featureDict in sliceFeatures])
        
            if feat in nonErodedFeatures.keys():
                nonErodedFeatures[feat].append(avgFeatureValue)
            else:
                nonErodedFeatures[feat] = [avgFeatureValue]
       
    if len(erodedSliceFeatures) > 0:
       for feat in sliceFeatures[0].keys():
           avgFeatureValue = np.mean([float(featureDict[feat]) for featureDict in erodedSliceFeatures]) 
           if feat in erodedFeatures.keys():
               erodedFeatures[feat].append(avgFeatureValue)
           else:
               erodedFeatures[feat] = [avgFeatureValue]
               

# Get correlation coefficients and save to dataframe
erosionCorrelation = pd.DataFrame(columns = ["feature", "pearson", "p-value"])
for feat in erodedFeatures.keys():
    
    nonErodedFeatureValues = np.array(nonErodedFeatures[feat])
    erodedFeatureValues = np.array(erodedFeatures[feat])
    
    # Drop nan values and normalize
    nan = np.isnan(nonErodedFeatureValues) | np.isnan(erodedFeatureValues)
    nonErodedFeatureValues = normalize(nonErodedFeatureValues[~nan])
    erodedFeatureValues = normalize(erodedFeatureValues[~nan])
    
    pearson = pearsonr(nonErodedFeatureValues, erodedFeatureValues)
    
    erosionCorrelation.loc[len(erosionCorrelation)] = [feat, pearson.statistic, pearson.pvalue]

erosionCorrelation.to_csv(os.path.join(outPath, "erosionRobustFeatures.csv"), index = False)