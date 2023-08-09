"""
This script runs a radiomic feature extraction for the BAP1 CT Images. The 
script is configured to run on data stored in a specific hierarchical format:
    Data Folder (i.e. TextureAnalysisFinal):
    --> Case
        --> Original Images (DICOM)
        --> Resampled Original Images (DICOM)
        --> Segmented Thorax Images (DICOM)
        --> Resampled Segmented Thorax Images (DICOM)
        --> Preprocessed Images (TIF)
        --> Resampled Preprocessed Images (TIF)
        --> Segmentation Probability Maps (TIF)
        --> Resampled Probability Maps (TIF)
        --> Resampled Segmentation Masks (TIF)
        
All the practical parts of feature extraction such as reading and manipulating 
files are here, whereas the specifics of the actual feature extraction process
are included in FeatureExtractor.py.

INPUT: Each stage of feature extraction (including which images to use, which 
preprocessing steps to take, which features to extract, etc.) is configurable 
using the configuration arguments specified below, to avoid manually adjusting 
code for each experiment. The configurations can be specified in a txt file, 
as in the configs/ folder.

OUTPUT: The script will output a CSV tabulating each input case as a row and 
each feature extracted as a column.

Written by Abbas Shaikh, Summer 2023
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pandas as pd
import yaml
import configargparse
import radiomics
import SimpleITK as sitk
from FeatureExtractor import FeatureExtractor
import warnings

# Silence warnings for PyRadiomics and PyFeats
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

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

# For reading booleans from config file
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

### Getting experiment configuration
parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True)

### Paths to read data and write experiment outputs
parser.add_argument('--data_path', type=str,
                    help = 'Path to folder containing images and masks.')
parser.add_argument('--save_path', type=str,
                    help = 'Path to save feature extraction output.')
parser.add_argument('--experiment_name', type=str,
                    help = 'Name of experiment (will be the title of experiment folder).')

### Customize Feature Extraction

# Feature aggregation
parser.add_argument('--aggregate_across_slices', type=str2bool, default = True,
                    help = 'Average texture features across slices, or report for one (represenative) slice only')

# Specifying features to extract
parser.add_argument('--shape_features', type=str2bool, default = True,
                    help = 'Extract shape features?')
parser.add_argument('--intensity_features', type=str2bool, default = True,
                    help = 'Extract intensity features?')
parser.add_argument('--radiomic_features', type=str2bool, default = True,
                    help = 'Extract radiomic texture features?')
parser.add_argument('--radiomic_feature_classes', type=str, nargs = "+", default = ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'gldzm', 'ngldm'],
                    help = 'Texture features to extract via radiomics feature extractor')
parser.add_argument('--laws_features', type=str2bool, default = True,
                    help = 'Extract Laws features?')
parser.add_argument('--fractal_features', type=str2bool, default = True,
                    help = 'Extract fractal dimension features?')
parser.add_argument('--fourier_features', type=str2bool, default = True,
                    help = 'Extract Fourier power spectrum features?')

### Customize Preprocessing Pipeline

# Specifying which CT images to use
parser.add_argument('--segmented_thorax', type=str2bool, default = True,
                    help = 'Use thorax-segmented CT scans instead of raw, unprocessed CT scans')
parser.add_argument('--windowing', type=str2bool, default = False,
                    help = 'Use thorax-segmented CT scans after preprocessing with windowing operation applied to enhance contrast')
parser.add_argument('--resampled_slice_thickness', type=str2bool, default = True,
                    help = 'Use images and masks after resampling to standardized slice thickness (3mm)')

# Specifying which tumor segmentations to use
parser.add_argument('--corrected_contours', type=str2bool, default = False,
                    help = 'Use radiologist corrected tumor contours rather than raw output of segmentation CNN')
parser.add_argument('--threshold', type=float, default = 0.01,
                    help = 'Threshold for segmentation probability map')

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

# Parameters for getting images from eroded masks
parser.add_argument('--rotate', type=str2bool, default = False,
                    help = 'Rotate image and mask prior to extracting features?')
parser.add_argument('--rotate_range', type=float, nargs = 2, default = [-15, 15],
                    help = 'Range of angles (degrees) within which to randomly rotate the image and mask')

parser.add_argument('--adapt_size', type=str2bool, default = False,
                    help = 'Perform volume adaptation, i.e. erode or dilate mask prior to extracting features?')
parser.add_argument('--adapt_range', type=int, nargs = 2, default = [-2, 2],
                    help = 'Range or radius within which to randomly erode/dilate mask')

parser.add_argument('--randomize_contours', type=str2bool, default = False,
                    help = 'Randomize maks contours?')

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
    
    try:
    
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
    
    # Handling keyboard interrupts during feature extraction
    except KeyboardInterrupt:
        print("Stopping Feature Extraction.")
        break

# Create and export dataframe output
allFeatures = pd.DataFrame.from_dict(allFeatures, orient = 'columns')
allFeatures.insert(0, 'Case', processedCases)

outCSV = os.path.join(experimentPath, "features.csv")
print("Saving output to:", outCSV)
allFeatures.to_csv(outCSV, index = False)