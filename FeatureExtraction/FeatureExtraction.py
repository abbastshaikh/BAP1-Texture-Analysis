import os
import pandas as pd
import yaml
import configargparse
import radiomics
import SimpleITK as sitk
import numpy as np
from FeatureExtractionUtil import FeatureExtractor, str2bool

# Set random seed
np.random.seed(0)

# Silence warnings for PyRadiomics and PyFeats
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

import warnings
# Warning for Laws features extraction associated with sparse segmentation labels
warnings.filterwarnings(
    action = 'ignore',
    category = RuntimeWarning,
    module = 'pyfeats',
    message = 'invalid value encountered in scalar divide'
)

# Warning for fractal features extraction associated with sparse segmentation labels
warnings.filterwarnings(
    action = 'ignore',
    category = RuntimeWarning,
    module = 'pyfeats',
    message = 'divide by zero encountered in log10'
)

### Getting experiment configuration
parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True)

### Paths to read data and write experiment outputs
parser.add_argument('--data_path', type=str,
                    help = 'Path to folder containing images')
parser.add_argument('--save_path', type=str,
                    help = 'Path to experiment outputs folder')
parser.add_argument('--experiment_name', type=str,
                    help = 'Name of experiment (will be title of experiment folder)')

### Customize Feature Extraction

# High-level approach (How to collect and aggregate texture featuress)
parser.add_argument('--approach', type=str, choices = ['full', 'each', 'patch'], default = 'full',
                    help = 'Approach for feature extraction(full = Full Image, each = Each Contiguous ROI, patch = Extract a Patch')
parser.add_argument('--aggregate_across_slices', type=str2bool, default = True,
                    help = 'Average texture features across slices, or report for one slice only')

# Specifying features to extract
parser.add_argument('--intensity_features', type=str2bool, default = True,
                    help = 'Extract intensity features - (y/n)')
parser.add_argument('--radiomic_features', type=str2bool, default = True,
                    help = 'Extract radiomic texture features - (y/n)')
parser.add_argument('--radiomic_feature_classes', type=str, nargs = "+", default = ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'gldzm', 'ngldm'],
                    help = 'Texture features to extract via radiomics feature extractor')
parser.add_argument('--laws_features', type=str2bool, default = True,
                    help = 'Extract Laws features - (y/n)')
parser.add_argument('--fractal_features', type=str2bool, default = True,
                    help = 'Extract fractal dimension features - (y/n)')
parser.add_argument('--fourier_features', type=str2bool, default = True,
                    help = 'Extract Fourier power spectrum features - (y/n)')

# Approach-specific parameters
parser.add_argument('--minimum_nonzero', type=int, default = 100,
                    help = 'Minimum number of nonzero-pixels in ROI mask to be considered (only for each method)')
parser.add_argument('--patch_size', type=int, default = 16,
                    help = 'Size of patch ROI to analyze (only for patch method)')

### Customize Preprocessing Pipeline

# Specifying which CT images to use
parser.add_argument('--segmented_thorax', type=str2bool, default = True,
                    help = 'Use thorax-segmented CT scans instead of raw, unprocessed CT scans')
parser.add_argument('--windowing', type=str2bool, default = False,
                    help = 'Use thorax-segmented CT scans after preprocessing with windowing operation applied to enhance contrast')
parser.add_argument('--resampled_slice_thickness', type=str2bool, default = True,
                    help = 'Uses images and masks after resampling to standardized slice thickness (3mm)')

# Specifying which tumor segmentations to use
parser.add_argument('--corrected_contours', type=str2bool, default = False,
                    help = 'Use radiologist corrected tumor contours rather than raw output of segmentation CNN')
parser.add_argument('--threshold', type=float, default = 0.01,
                    help = 'Threshold for segmentation probability map')
parser.add_argument('--mask_opening', type=str2bool, default = False,
                    help = 'Apply opening morphological operation to mask')

# Image standardization parameters (resampling and discretization)
parser.add_argument('--pixel_spacing', type=float, default = 0.,
                    help = 'Resample image and mask to standard pixel spacing. Set to 0 to not resample.')
parser.add_argument('--discretize', type=str, default = "", choices = ["", "fixedwidth", "fixedcount"],
                    help = 'Discretize gray levels of image (fixedwidth = fixed bin widths, fixedcount = fixed number of bins')
parser.add_argument('--n_bins', type=int, default = 32,
                    help = 'Number of bins to discretize gray levels of image')
parser.add_argument('--bin_width', type=float, default = 25,
                    help = 'Bin width to discretize gray levels of image')

# Preprocessing filter parameters
parser.add_argument('--preprocessing_filters', type=str, nargs = "+", choices = ['LoG', 'wavelet', 'LBP'], default = [],
                    help = 'Filter to apply to image prior to texture analysis. Choices are Laplacian of Gaussian (LoG), wavelet, or Local Binary Pattern (LBP)')
parser.add_argument('--LoG_sigma', type=float, nargs = "+", default = 2.,
                    help = 'Std of Gaussian kernel for LoG filter, can input multiple.')
parser.add_argument('--wavelet', type=str, nargs = "+", default = "bior1.1",
                    help = 'Wavelet to use for wavelet transform, can input multiple. Must be in pyWavelet.wavelist().')
parser.add_argument('--LBP_radius', type=float, nargs = "+", default = 1.,
                    help = 'Radius of Local Binary Pattern filter, can input multiple.')

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
    
    # Setting correct data paths based on experiment configuration
    if args.windowing:
        imgDir = "PreprocessedImgs"
    elif args.segmented_thorax:
        imgDir = "SegmentedThorax"
    else:
        imgDir = "OriginalImgs"
    
    if args.corrected_contours:
        labelsDir = "Masks"
    else:
        labelsDir = "prob_maps"
        
    if args.resampled_slice_thickness:
        imgDir = imgDir + "_Resampled"
        labelsDir = labelsDir + "_Resampled"
    
    # Get paths to images and labels
    imgsPath = sorted(os.listdir(os.path.join(args.data_path, cases[caseIdx], imgDir)))
    labelsPath = sorted(os.listdir(os.path.join(args.data_path, cases[caseIdx], labelsDir)))
    
    # Make sure we have the same number of images and segmenations
    assert len(imgsPath) == len(labelsPath)
    
    # Limit paths to only one slice if not aggregating across all slices
    if not args.aggregate_across_slices:
        imgsPath = [imgsPath[1]]
        labelsPath = [labelsPath[1]]
    
    # List to store all slices of images and segmentations for current case
    images = []
    labels = []
    
    # Iterate over each slice per case
    for idx in range(len(imgsPath)):
            
        imgPath = os.path.join(args.data_path, cases[caseIdx], imgDir, imgsPath[idx])
        labelPath = os.path.join(args.data_path, cases[caseIdx], labelsDir, labelsPath[idx])
        
        # Read in image
        image = sitk.ReadImage(imgPath)
        
        # Remove third dimension of slice if it exists
        if len(image.GetSize()) == 3:
            image = image[:, :, 0]
        
        # Windowed images are TIFs, which does not preserve image metadata (e.g. spacing)
        # We load DICOM image and transfer image metadata parameters here
        if args.windowing:
            if args.resampled_slice_thickness:
                dicom = sitk.ReadImage(os.path.join(args.data_path, cases[caseIdx], "SegmentedThorax_Resampled", 
                                                    imgsPath[idx].split(".")[0]))[:, :, 0] # Dicom has the same name as the TIF without the file extension
            else:
                dicom = sitk.ReadImage(os.path.join(args.data_path, cases[caseIdx], "SegmentedThorax", 
                                                    imgsPath[idx].split(".")[0]))[:, :, 0]
            
            image.CopyInformation(dicom)
        
        # Handling edge case, original image size doesn't match
        if caseName == "29_18000101_CT_AXL_W":
            image = image[:, -512:]
        
        # Read in segmentation
        label = sitk.ReadImage(labelPath)
        
        # Masks are stored as 0 as negative, 255 as positive, so we rescale to 0 and 1
        if args.corrected_contours:
            label = sitk.RescaleIntensity(label, outputMinimum = 0, outputMaximum = 1)
            label[0, 0] = 0
        # Otherwise we binarize at specified threshold
        else:
            label = sitk.BinaryThreshold(label, lowerThreshold = args.threshold)
        
        # Add to list of images/segmentations
        images.append(image)
        labels.append(label)
        
    # Extract features, save if successfull
    extractedFeatures = featureExtractor.extractFeatures(images, labels)
    
    if extractedFeatures:
        allFeatures.append(extractedFeatures)
        processedCases.append(caseName)
    
# Create and export dataframe output
allFeatures = pd.DataFrame.from_dict(allFeatures, orient = 'columns')
allFeatures.insert(0, 'Case', processedCases)

allFeatures.to_csv(os.path.join(experimentPath, "features.csv"), index = False)