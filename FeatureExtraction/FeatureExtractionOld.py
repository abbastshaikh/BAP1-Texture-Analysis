import os
import pandas as pd
import yaml
import configargparse
import radiomics
import SimpleITK as sitk
import numpy as np
from FeatureExtractionUtilOld import extractFeatures, str2bool, preprocess

np.random.seed(0)
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

# Get experiment config
parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True)
parser.add_argument('--approach', type=str, choices = ['full', 'each', 'patch'], default = 'full',
                    help = 'Approach for feature extraction(full = Full Image, each = Each Contiguous ROI, patch = Extract a Patch')
parser.add_argument('--data_path', type=str,
                    help = 'Path to folder containing images')
parser.add_argument('--save_path', type=str,
                    help = 'Path to save texture features and experiment config')
parser.add_argument('--aggregate_across_slices', type=str2bool, default = True,
                    help = 'Average texture features across slices, or report for one slice only')

# Feature Extraction
parser.add_argument('--intensity_features', type=str2bool, default = True,
                    help = 'Extract intensity features - (y/n)')
parser.add_argument('--radiomic_features', type=str2bool, default = True,
                    help = 'Extract radiomic texture features - (y/n)')
parser.add_argument('--radiomic_feature_classes', type=str, nargs = "+", default = ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'gldzm', 'ngldm'],
                    help = 'Texture features to extract via radiomics feature extractor')
parser.add_argument('--laws_features', type=str2bool, default = True,
                    help = 'Extract Laws features - (y/n)')
parser.add_argument('--minimum_nonzero', type=int, default = 100,
                    help = 'Minimum number of nonzero-pixels in ROI mask to be considered (only for each method)')
parser.add_argument('--patch_size', type=int, default = 16,
                    help = 'Size of patch ROI to analyze (only for patch method)')

# Preprocessing
parser.add_argument('--pixel_spacing', type=float, default = 0.,
                    help = 'Resample image and mask to standard pixel spacing. Set to 0 to not resample.')
parser.add_argument('--discretize', type=str, default = "", choices = ["", "fixedwidth", "fixedcount"],
                    help = 'Discretize gray levels of image (fixedwidth = fixed bin widths, fixedcount = fixed number of bins')
parser.add_argument('--n_bins', type=int, default = 64,
                    help = 'Number of bins to discretize gray levels of image')
parser.add_argument('--bin_width', type=float, default = 25,
                    help = 'Bin width to discretize gray levels of image')

parser.add_argument('--mask_opening', type=str2bool, default = False,
                    help = 'Apply opening morphological operation to mask')

parser.add_argument('--preprocessing_filter', type=str, choices = ['', 'LoG'], default = '',
                    help = 'Filter to apply to image prior to texture analysis')
parser.add_argument('--LoG_sigma', type=float, default = 2.,
                    help = 'Std of Gaussian kernel for LoG filter')

parser.add_argument('--threshold', type=float, default = 0.01,
                    help = 'Threshold for segmentation probability map')

args = parser.parse_args()

args_dict = vars(args)

# Create experiment directory and save configs
os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, 'config.yaml'), 'w') as file:
    yaml.dump(args.__dict__, file, default_flow_style=False)

# Create dataframe to store texture features
allFeatures = []
processedCases = []
    
# Iterate over cases (each dir = case)
cases = os.listdir(args.data_path)
for caseIdx in range(len(cases)):
    
    caseName = cases[caseIdx].rsplit("_", 1)[0]
    print("Processing Case " + str(caseIdx + 1) + "/" + str(len(cases)) + ": ", caseName)
    
    originalImgsPath = os.listdir(os.path.join(args.data_path, cases[caseIdx], "OriginalImgs"))
    probMapsPath = os.listdir(os.path.join(args.data_path, cases[caseIdx], "prob_maps"))
    
    # Limit paths to only one slice if not averaging across all slices
    if not args.aggregate_across_slices:
        originalImgsPath = [originalImgsPath[1]]
        probMapsPath = [probMapsPath[1]]
    
    # List to store feature dictionaries for each slice
    sliceFeatures = []
    
    # Iterate over each slice per case
    for idx in range(len(originalImgsPath)):
        
        imgPath = os.path.join(args.data_path, cases[caseIdx], "OriginalImgs", originalImgsPath[idx])
        probPath = os.path.join(args.data_path, cases[caseIdx], "prob_maps", probMapsPath[idx])
        
        # Read in image
        image = sitk.ReadImage(imgPath)[:, :, 0]
        
        # Handling edge case, original image size doesn't match
        if caseName == "29_18000101_CT_AXL_W":
            image = image[:, -512:]
        
        # Read in probability map and threshold to produce mask
        probMap = sitk.ReadImage(probPath)
        mask = sitk.BinaryThreshold(probMap, lowerThreshold = args.threshold)
        
        # Preprocess image and mask via preprocessing pipline
        image, modifiedImage, mask = preprocess(image, mask,  
                                                args.pixel_spacing,
                                                args.discretize,
                                                args.n_bins,
                                                args.bin_width,
                                                args.preprocessing_filter,
                                                args.LoG_sigma,
                                                args.mask_opening)
        
        # Extract features. If features were successfully extracted, add to sliceFeatures
        featuresExtracted = extractFeatures(image, mask, args, radiomicsExtractor)
        if featuresExtracted:
            sliceFeatures.append(featuresExtracted)
        
    # Average features across all slices, don't add case if no features were successfully extracted
    if len(sliceFeatures) > 0:
        averageFeatures = {}
        for feat in sliceFeatures[0].keys():
            averageFeatures[feat] = np.mean([float(featureDict[feat]) for featureDict in sliceFeatures])
        allFeatures.append(averageFeatures)
        processedCases.append(caseName)
    else:
        allFeatures.extend(sliceFeatures)
        processedCases.extend([caseName] * len(sliceFeatures))
        
# Create and export dataframe output
allFeatures = pd.DataFrame.from_dict(allFeatures, orient = 'columns')
allFeatures.insert(0, 'Case', processedCases)

allFeatures.to_csv(os.path.join(args.save_path, "features.csv"), index = False)