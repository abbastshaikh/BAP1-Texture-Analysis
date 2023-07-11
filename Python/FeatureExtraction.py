import os
import pandas as pd
import yaml
import configargparse
import radiomics
import SimpleITK as sitk
import numpy as np
from FeatureExtractionUtil import FeatureExtractor, str2bool

np.random.seed(0)
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

# Get experiment config
parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True)
parser.add_argument('--data_path', type=str,
                    help = 'Path to folder containing images')
parser.add_argument('--save_path', type=str,
                    help = 'Path to experiment outputs folder')
parser.add_argument('--experiment_name', type=str,
                    help = 'Name of experiment (will be title of experiment folder)')

# Feature Extraction
parser.add_argument('--approach', type=str, choices = ['full', 'each', 'patch'], default = 'full',
                    help = 'Approach for feature extraction(full = Full Image, each = Each Contiguous ROI, patch = Extract a Patch')
parser.add_argument('--aggregate_across_slices', type=str2bool, default = True,
                    help = 'Average texture features across slices, or report for one slice only')

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
parser.add_argument('--threshold', type=float, default = 0.01,
                    help = 'Threshold for segmentation probability map')
parser.add_argument('--resampled_slice_thickness', type=str2bool, default = True,
                    help = 'Uses images and masks resampled to standardized slice thickness (3mm)')
parser.add_argument('--pixel_spacing', type=float, default = 0.,
                    help = 'Resample image and mask to standard pixel spacing. Set to 0 to not resample.')
parser.add_argument('--mask_opening', type=str2bool, default = False,
                    help = 'Apply opening morphological operation to mask')

parser.add_argument('--discretize', type=str, default = "", choices = ["", "fixedwidth", "fixedcount"],
                    help = 'Discretize gray levels of image (fixedwidth = fixed bin widths, fixedcount = fixed number of bins')
parser.add_argument('--n_bins', type=int, default = 64,
                    help = 'Number of bins to discretize gray levels of image')
parser.add_argument('--bin_width', type=float, default = 25,
                    help = 'Bin width to discretize gray levels of image')

parser.add_argument('--preprocessing_filter', type=str, choices = ['', 'LoG'], default = '',
                    help = 'Filter to apply to image prior to texture analysis')
parser.add_argument('--LoG_sigma', type=float, default = 2.,
                    help = 'Std of Gaussian kernel for LoG filter')

args = parser.parse_args()
args_dict = vars(args)

# Create experiment directory and save configs
experimentPath = os.path.join(args.save_path, args.experiment_name)
os.makedirs(experimentPath, exist_ok=True)
with open(os.path.join(experimentPath, 'config.yaml'), 'w') as file:
    yaml.dump(args.__dict__, file, default_flow_style=False)
    
# Create feature extractor object
featureExtractor = FeatureExtractor(args_dict)

# Create dataframe to store texture features
allFeatures = []
processedCases = []
    
# Iterate over cases (each dir = case)
cases = os.listdir(args.data_path)
for caseIdx in range(len(cases)):
    
    caseName = cases[caseIdx].rsplit("_", 1)[0]
    print("Processing Case " + str(caseIdx + 1) + "/" + str(len(cases)) + ": ", caseName)
    
    if args.resampled_slice_thickness:
        imgDir = "ResampledImgs_3mm"
        probDir = "prob_maps_Resampled"
    else:
        imgDir = "OriginalImgs"
        probDir = "prob_maps"
    
    originalImgsPath = os.listdir(os.path.join(args.data_path, cases[caseIdx], imgDir))
    probMapsPath = os.listdir(os.path.join(args.data_path, cases[caseIdx], probDir))
    
    # Limit paths to only one slice if not aggregating across all slices
    if not args.aggregate_across_slices:
        originalImgsPath = [originalImgsPath[1]]
        probMapsPath = [probMapsPath[1]]
    
    # List to store all slices of images and probability maps for current case
    images = []
    probMaps = []
    
    # Iterate over each slice per case
    for idx in range(len(originalImgsPath)):
            
        imgPath = os.path.join(args.data_path, cases[caseIdx], imgDir, originalImgsPath[idx])
        probPath = os.path.join(args.data_path, cases[caseIdx], probDir, probMapsPath[idx])
        
        # Read in image
        image = sitk.ReadImage(imgPath)[:, :, 0]
        
        # Handling edge case, original image size doesn't match
        if caseName == "29_18000101_CT_AXL_W":
            image = image[:, -512:]
        
        # Read in probability map and threshold to produce mask
        probMap = sitk.ReadImage(probPath)
        
        # Add to list of images/prob maps
        images.append(image)
        probMaps.append(probMap)
        
    # Extract features, save if successfull
    extractedFeatures = featureExtractor.extractFeatures(images, probMaps)
    
    if extractedFeatures:
        allFeatures.append(extractedFeatures)
        processedCases.append(caseName)
    
# Create and export dataframe output
allFeatures = pd.DataFrame.from_dict(allFeatures, orient = 'columns')
allFeatures.insert(0, 'Case', processedCases)

allFeatures.to_csv(os.path.join(experimentPath, "features.csv"), index = False)