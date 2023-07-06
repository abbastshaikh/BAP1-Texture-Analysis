import configargparse
import radiomics
import pyfeats
import SimpleITK as sitk
import cv2
import numpy as np

# For reading booleans from text file
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
        
# Resample 2-D image to new pixel spacing
def resampleToSpacing (image, mask, spacing):
    # Initialize resampler
    resampler = sitk.ResampleImageFilter()
    
    resampler.SetOutputSpacing((spacing, spacing))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    # Calculate new size using new spacing
    newSize = [int(size * (oldSpacing / newSpacing)) for 
               size, newSpacing, oldSpacing in zip(image.GetSize(), resampler.GetOutputSpacing(), image.GetSpacing())]
    resampler.SetSize(newSize)
    
    # Linear Interpolation for Image
    resampler.SetInterpolator(sitk.sitkLinear)
    resampledImage = resampler.Execute(image)
     
    # Nearest Neighbor Interpolation for Mask
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampledMask = resampler.Execute(mask)
    
    return resampledImage, resampledMask
    
# Preprocessing pipeline
def preprocess (image, mask, 
                pixel_spacing,
                discretize,
                n_bins,
                bin_width,
                preprocessing_filter,
                LoG_sigma,
                mask_opening):
    
    # Match mask geometry to image geometry
    mask.SetDirection(image.GetDirection())
    mask.SetSpacing(image.GetSpacing())
    mask.SetOrigin(image.GetOrigin())
    
    # Normalize to standard pixel spacing
    if pixel_spacing:
        image, mask = resampleToSpacing(image, mask, pixel_spacing)
        
    # Discretize gray levels
    if discretize:
        # Get numpy arrays of image
        parameterMatrix = sitk.GetArrayFromImage(image)
        parameterMatrixCoordinates = np.nonzero(sitk.GetArrayFromImage(mask))
        
        # Discretize gray levels
        if discretize == "fixedwidth":
            discretizedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                  parameterMatrixCoordinates,
                                                                  binWidth = bin_width
                                                                  )[0]
            
        elif discretize == "fixedcount":
            discretizedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                  parameterMatrixCoordinates,
                                                                  binCount = n_bins
                                                                  )[0]
    
        # Convert back to SimpleITK image, and match geometry
        discretizedImage = sitk.GetImageFromArray(discretizedImage)
        discretizedImage.SetDirection(image.GetDirection())
        discretizedImage.SetSpacing(image.GetSpacing())
        discretizedImage.SetOrigin(image.GetOrigin())
        image = discretizedImage
        
    # Apply preprocessing image filters
    if preprocessing_filter == "LoG":
        image = sitk.LaplacianRecursiveGaussian(image, sigma = LoG_sigma)
        
    # Apply morphological operations to mask
    if mask_opening:
        mask = sitk.BinaryMorphologicalOpening(mask)
        
    return image, mask
    

# Generates texture features from image and mask
def getTextureFeatures (image, mask, 
                        radiomic_features, 
                        laws_features,
                        radiomicsExtractor = None):
    features = {}
    if radiomic_features:
        features.update(dict(radiomicsExtractor.computeFeatures(image, mask, "original")))
    if laws_features:
        l_features, l_labels = pyfeats.lte_measures(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(mask), l = 5)
        features.update(dict(zip(l_labels, l_features)))
    return features

# Generate texture features from a full image
def fullImageFeatureExtraction (image, mask,
                                radiomic_features,
                                laws_features,
                                radiomicsExtractor = None):
    # Crop image/mask to tumor region
    boundingBox = radiomics.imageoperations.checkMask(image, mask)[0]
    croppedImage, croppedMask = radiomics.imageoperations.cropToTumorMask(image, mask, boundingBox)
    
    return getTextureFeatures(croppedImage, croppedMask, 
                              radiomic_features, 
                              laws_features, 
                              radiomicsExtractor)

# Generate texture features from contiguous ROIs
def individualROIFeatureExtraction (image, mask, 
                                    minimum_nonzero,
                                    radiomic_features,
                                    laws_features,
                                    radiomicsExtractor = None):
    
    maskArray = sitk.GetArrayFromImage(mask)
    
    # Get contours of all ROIs
    contours = cv2.findContours(maskArray, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_NONE)[0]
    
    # Get features from each ROI
    roiFeatures = []
    for contour in contours:
        boundingBox = [np.min(contour[:, :, 0]), 
                       np.max(contour[:, :, 0]), 
                       np.min(contour[:, :, 1]), 
                       np.max(contour[:, :, 1])]
        
        # If ROI is large enough, get features
        if np.count_nonzero(maskArray[boundingBox[2]:boundingBox[3], boundingBox[0]:boundingBox[1]]) >= minimum_nonzero:
            croppedImage = image[boundingBox[0]:boundingBox[1], boundingBox[2]:boundingBox[3]]
            croppedMask = mask[boundingBox[0]:boundingBox[1], boundingBox[2]:boundingBox[3]]
            roiFeatures.append(getTextureFeatures(croppedImage, croppedMask, 
                                                  radiomic_features, 
                                                  laws_features, 
                                                  radiomicsExtractor))
    
    # Average features from all ROIs
    averageROIFeatures = {}
    if len(roiFeatures) > 0:
        for feat in roiFeatures[0].keys():
            averageROIFeatures[feat] = np.mean([float(featureDict[feat]) for featureDict in roiFeatures])
    
    return averageROIFeatures

# Generate texture features from a patch ROI sampled from the image
def patchROIFeatureExtraction (image, mask, 
                               patch_size, 
                               radiomic_features, 
                               laws_features,
                               radiomicsExtractor = None):
    
    maskArray = sitk.GetArrayFromImage(mask)
    
    # Get indices of nonzero pixels in mask
    nonzero = np.array(np.nonzero(maskArray)).T
    
    # Randomize order so we don't prefer pixels closer to origin
    np.random.shuffle(nonzero)
   
    # Iterate till we find patch
    for point in nonzero:
        
        # Check if patch is in bounds
        if point[0] < maskArray.shape[0] - patch_size and point[1] < maskArray.shape[1] - patch_size:
            
            # Check if patch is all tumor pixels
            if np.sum(maskArray[point[0]:point[0] + patch_size, point[1]:point[1] + patch_size]) == patch_size ** 2:
                
                imagePatch = image[point[1]:point[1] + patch_size, point[0]:point[0] + patch_size]
                maskPatch = mask[point[1]:point[1] + patch_size, point[0]:point[0] + patch_size]

                return getTextureFeatures(imagePatch, maskPatch, 
                                          radiomic_features, 
                                          laws_features, 
                                          radiomicsExtractor)
    
    # Return empty dictionary if no patch is found
    return {}
 
# Generic feature extractor wrapper    
def extractFeatures (image, mask, args, radiomicsExtractor = None):       
    if args.approach == "full":
        return fullImageFeatureExtraction(image, mask,
                                          radiomic_features = args.radiomic_features,
                                          laws_features = args.laws_features,
                                          radiomicsExtractor = radiomicsExtractor)
    
    if args.approach == "each":
        return individualROIFeatureExtraction(image, mask, 
                                              minimum_nonzero = args.minimum_nonzero,
                                              radiomic_features = args.radiomic_features,
                                              laws_features = args.laws_features,
                                              radiomicsExtractor = radiomicsExtractor)
    
    if args.approach == "patch":
        return patchROIFeatureExtraction(image, mask, 
                                         patch_size = args.patch_size,
                                         radiomic_features = args.radiomic_features,
                                         laws_features = args.laws_features,
                                         radiomicsExtractor = radiomicsExtractor)