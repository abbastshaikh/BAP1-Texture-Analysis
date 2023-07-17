import os
import pandas as pd
import radiomics
import SimpleITK as sitk
import numpy as np
from FeatureExtractionUtil import FeatureExtractor
from scipy.stats import pearsonr


import matplotlib.pyplot as plt



np.random.seed(0)
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

# Setting config for extraction pipeline
args = {
    "data_path": "D:\BAP1\Data\TextureAnalysisFinal",
    "save_path": "D:\BAP1\Experiments",
    "experiment_name": "ErosionRobustFeatures",
    "approach": "full",
    "aggregate_across_slices": True,
    "intensity_features": True,
    "radiomic_features": True,
    "radiomic_feature_classes": ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'gldzm', 'ngldm'],
    "laws_features": True,
    "segmented_thorax": True,
    "windowing": False,
    "resampled_slice_thickness": True,
    "corrected_contours": True,
    "threshold": 0.01,
    "mask_opening": False,
    "pixel_spacing": 0.75,
    "discretize": 'fixedcount',
    "n_bins": 32,
    "preprocessing_filter": ''
    }

# Create experiment directory and save configs
experimentPath = os.path.join(args["save_path"], args["experiment_name"])
os.makedirs(experimentPath, exist_ok=True)

# Create feature extractor object
featureExtractor = FeatureExtractor(args)

# Create dataframe to store texture features
allFeatures = []
allErodedFeatures = []
    
# Iterate over cases (each dir = case)
cases = os.listdir(args["data_path"])
for caseIdx in range(len(cases)):
    
    caseName = cases[caseIdx].rsplit("_", 1)[0]
    print("Processing Case " + str(caseIdx + 1) + "/" + str(len(cases)) + ": ", caseName)
    
    # Setting correct data paths based on experiment configuration
    if args["windowing"]:
        imgDir = "PreprocessedImgs"
    elif args["segmented_thorax"]:
        imgDir = "SegmentedThorax"
    else:
        imgDir = "OriginalImgs"
    
    if args["corrected_contours"]:
        labelsDir = "Masks"
    else:
        labelsDir = "prob_maps"
        
    if args["resampled_slice_thickness"]:
        imgDir = imgDir + "_Resampled"
        labelsDir = labelsDir + "_Resampled"
    
    # Get paths to images and labels
    imgsPath = sorted(os.listdir(os.path.join(args["data_path"], cases[caseIdx], imgDir)))
    labelsPath = sorted(os.listdir(os.path.join(args["data_path"], cases[caseIdx], labelsDir)))
    
    # Make sure we have the same number of images and segmenations
    assert len(imgsPath) == len(labelsPath)
    
    # Limit paths to only one slice if not aggregating across all slices
    if not args["aggregate_across_slices"]:
        imgsPath = [imgsPath[1]]
        labelsPath = [labelsPath[1]]
    
    # List to store all slices of images and segmentations for current case
    images = []
    labels = []
    erodedLabels = []
    
    # Iterate over each slice per case
    for idx in range(len(imgsPath)):
            
        imgPath = os.path.join(args["data_path"], cases[caseIdx], imgDir, imgsPath[idx])
        labelPath = os.path.join(args["data_path"], cases[caseIdx], labelsDir, labelsPath[idx])
        
        # Read in image
        image = sitk.ReadImage(imgPath)
        
        # Remove third dimension of slice if it exists
        if len(image.GetSize()) == 3:
            image = image[:, :, 0]
        
        # Windowed images are TIFs, which does not preserve image geometry (e.g. spacing)
        # We load DICOM image and transfer image geometry parameters here
        if args["windowing"]:
            if args["resampled_slice_thickness"]:
                dicom = sitk.ReadImage(os.path.join(args["data_path"], cases[caseIdx], "SegmentedThorax_Resampled", 
                                                    imgsPath[idx].split(".")[0]))[:, :, 0] # Dicom has the same name as the TIF without the file extension
            else:
                dicom = sitk.ReadImage(os.path.join(args["data_path"], cases[caseIdx], "SegmentedThorax", 
                                                    imgsPath[idx].split(".")[0]))[:, :, 0]
            
            image.SetDirection(dicom.GetDirection())
            image.SetSpacing(dicom.GetSpacing())
            image.SetOrigin(dicom.GetOrigin())
        
        # Handling edge case, original image size doesn't match
        if caseName == "29_18000101_CT_AXL_W":
            image = image[:, -512:]
        
        # Read in segmentation and binarize, if necessary
        label = sitk.ReadImage(labelPath)
        
        # Masks are stored as 0 as negative, 255 as positive, so we rescale to 0 and 1
        if args["corrected_contours"]:
            label = sitk.RescaleIntensity(label, outputMinimum = 0, outputMaximum = 1)
        # Otherwise we binarize at specified threshold
        else:
            label = sitk.BinaryThreshold(label, lowerThreshold = args["threshold"])
        
        # Get texture features for eroded masks
        erodeFilter = sitk.BinaryErodeImageFilter()
        erodeFilter.SetKernelRadius(1)
        erodedLabel = erodeFilter.Execute(label)
        
        # Add to lists of images/segmentations
        images.append(image)
        labels.append(label)
        erodedLabels.append(erodedLabel)
        
    # Extract features, save if successfull
    extractedFeatures = featureExtractor.extractFeatures(images, labels)
    extractedErodedFeatures = featureExtractor.extractFeatures(images, erodedLabels)
    
    if extractedFeatures:
        allFeatures.append(extractedFeatures)
        allErodedFeatures.append(extractedErodedFeatures)
    
# Convert to dataframes
allFeatures = pd.DataFrame.from_dict(allFeatures, orient = 'columns')
allErodedFeatures = pd.DataFrame.from_dict(allErodedFeatures, orient = 'columns')

# Normalize to zero mean and unit variance (numpy)
def normalize (data):
    return (data - data.mean()) / data.std()

# Get correlation coefficients and save to dataframe
featureCorrelation = pd.DataFrame(columns = ["feature", "pearson", "p-value"])
for feat in allFeatures.columns:
    
    featureValues = allFeatures[feat]
    erodedFeatureValues = allErodedFeatures[feat]
    
    # Drop cases with null values, normalize remaining feature values
    dropCases = featureValues.isnull() | erodedFeatureValues.isnull()
    featureValues = normalize(np.array(featureValues[~dropCases]))
    erodedFeatureValues = normalize(np.array(erodedFeatureValues[~dropCases]))
    
    # Calculate and save correlation values
    pearson = pearsonr(featureValues, erodedFeatureValues)
    featureCorrelation.loc[len(featureCorrelation)] = [feat, pearson.statistic, pearson.pvalue]
    
allFeatures.to_csv(os.path.join(experimentPath, "features.csv"), index = False)
allErodedFeatures.to_csv(os.path.join(experimentPath, "featuresEroded.csv"), index = False)
featureCorrelation.to_csv(os.path.join(experimentPath, "featureCorrelation.csv"), index = False)