import configargparse
import radiomics
import pyfeats
import SimpleITK as sitk
import numpy as np
from nyxus import Nyxus

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
    
class FeatureExtractor:
    
    def __init__ (self, args_dict):
        
        # Save input arguments
        self.args = args_dict
        
        # Create feature extractor objects
        if self.args["shape_features"]:
            self.shapeExtractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
            self.shapeExtractor.disableAllFeatures()
            self.shapeExtractor.enableFeatureClassByName('shape2D')
        
        if self.args["intensity_features"]:
            self.intensityExtractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
            self.intensityExtractor.disableAllFeatures()
            self.intensityExtractor.enableFeatureClassByName('firstorder')
            
        if self.args["radiomic_features"]: 
            self.pyRadiomicsFeatures = [arg for arg in self.args["radiomic_feature_classes"] if arg in ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']]
            self.nyxusFeatures = [arg.upper() for arg in self.args["radiomic_feature_classes"] if arg in ['gldzm', 'ngldm']]
            
            self.pyRadiomicsExtractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
            self.pyRadiomicsExtractor.disableAllFeatures()
            for feature_class in self.pyRadiomicsFeatures:
                self.pyRadiomicsExtractor.enableFeatureClassByName(feature_class)
        
            self.nyxusExtractor = Nyxus(["*ALL*"])
            
    # Preprocess single SimpleITK image and segmentation mask and return resampled image 
    # and mask and processed image after applying discretization and preprocessing filters
    def preprocess (self, image, mask):
        
        # Match mask metadata to image metadata
        mask.CopyInformation(image)
        
        # Resample to standardized pixel spacing
        if self.args["pixel_spacing"]:
            resampledImage, resampledMask = FeatureExtractor.resampleToSpacing(image, mask, self.args["pixel_spacing"])
                
        # Apply morphological operations to mask
        if self.args["mask_opening"]:
            resampledMask = sitk.BinaryMorphologicalOpening(resampledMask)
            
        # Apply preprocessing image filters
        filteredImages = []
        filterNames = []
        
        # Laplacian of Gaussian filter
        if "LoG" in self.args["preprocessing_filters"]:
            for sigma in self.args["LoG_sigma"]:
                LoGImage = sitk.LaplacianRecursiveGaussian(resampledImage, sigma = sigma)
                filteredImages.append(LoGImage)
                filterNames.append("LoG_sigma=" + str(sigma))
            
        # Wavelet filter
        if "wavelet" in self.args["preprocessing_filters"]:
            for wavelet in self.args["wavelet"]:
                waveletTransform = radiomics.imageoperations.getWaveletImage(resampledImage, resampledMask, wavelet = wavelet)
                
                # Iterating through all decompositions
                while True:
                    try:
                        waveletImage = next(waveletTransform)
                        filteredImages.append(waveletImage[0])
                        filterNames.append(waveletImage[1].replace("wavelet", wavelet))
                    except StopIteration:
                        break
        
        # Local Binary Pattern filter
        if "LBP" in self.args["preprocessing_filters"]:
            for radius in self.args["LBP_radius"]:
                LBPImage = next(radiomics.imageoperations.getLBP2DImage(resampledImage, resampledMask, lbp2DRadius = radius))[0]
                filteredImages.append(LBPImage)
                filterNames.append("LBP_radius=" + str(radius))
        
        return resampledImage, filteredImages, filterNames, resampledMask
    
    # Preprocess list of image and segmentation mask slices
    def preprocessMultiple (self, images, masks):
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. Preprocessing failed.")
            return
        
        # Compiled resampled and preprocessed images and masks across multiple slices
        resampledImages = []
        filteredImages_ = []
        resampledMasks = []
        
        for idx in range(len(images)):
            resampledImage, filteredImages, filterNames, resampledMask = self.preprocess(images[idx], masks[idx])
            
            resampledImages.append(resampledImage)
            filteredImages_.append(filteredImages)
            resampledMasks.append(resampledMask)
            
        return resampledImages, filteredImages_, filterNames, resampledMasks
    
    # Gets shape features from single SimpleITK image and mask
    def getShapeFeatures (self, image, mask):
        
        features = {}
        
        # Extracts intensity features via PyRadiomics
        if self.args["shape_features"]:
            boundingBox = radiomics.imageoperations.checkMask(image, mask)[0]
            features.update(dict(self.shapeExtractor.computeShape(image, mask, boundingBox)))

        return features
    
    # Gets intensity features from single SimpleITK image and mask
    def getIntensityFeatures (self, image, mask):
        
        features = {}
        
        # Extracts intensity features via PyRadiomics
        if self.args["intensity_features"]:
            features.update(dict(self.intensityExtractor.computeFeatures(image, mask, "original")))

        return features
        
    # Gets texture features from single SimpleITK image and mask
    # Can perform discretization prior to feature extraction
    def getTextureFeatures (self, image, mask, discretize = False):
        
        # Discretize gray levels
        if discretize:
            # Get numpy arrays of image
            parameterMatrix = sitk.GetArrayFromImage(image)
            parameterMatrixCoordinates = np.nonzero(sitk.GetArrayFromImage(mask))
            
            # Discretize gray levels to fixed bin width/size
            if self.args["discretize"] == "fixedwidth":
                processedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                      parameterMatrixCoordinates,
                                                                      binWidth = self.args["bin_width"]
                                                                      )[0]
             
            # Discretize gray levels to fixed bin number/count
            elif self.args["discretize"] == "fixedcount":
                processedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                      parameterMatrixCoordinates,
                                                                      binCount = self.args["n_bins"]
                                                                      )[0]
        
            # Convert back to SimpleITK image, and match metadata
            processedImage = sitk.GetImageFromArray(processedImage)
            processedImage.CopyInformation(image)
            
        else:
            processedImage = image
        
        features = {}
        
        # Converts SimpleITK images to arrays for some packages
        imgArray = sitk.GetArrayFromImage(processedImage)
        maskArray = sitk.GetArrayFromImage(mask)
        
        # Extracts radiomic features via PyRadiomics and Nyxus
        if self.args["radiomic_features"]:
            if self.pyRadiomicsFeatures:
                features.update(dict(self.pyRadiomicsExtractor.computeFeatures(processedImage, mask, "original")))
            if self.nyxusFeatures:
                ### Nyxus outputs dataframe, converts to dictionary
                n_features = self.nyxusExtractor.featurize(imgArray, maskArray).to_dict('list')
                n_features = {k: v[0] for k, v in n_features.items() if k.startswith(tuple(self.nyxusFeatures))}
                features.update(n_features)
                
        # Extracts Laws' features via PyFeats
        if self.args["laws_features"]:
            laws_features, laws_labels = pyfeats.lte_measures(imgArray, maskArray, l = 5)
            features.update(dict(zip(laws_labels, laws_features)))
            
        # Extracts Laws' features via PyFeats
        if self.args["fractal_features"]:
            fractal_features, fractal_labels = pyfeats.fdta(imgArray, maskArray)
            features.update(dict(zip(fractal_labels, fractal_features)))
            
        # Extracts Laws' features via PyFeats
        if self.args["fourier_features"]:
            fourier_features, fourier_labels = pyfeats.fps(imgArray, maskArray)
            features.update(dict(zip(fourier_labels, fourier_features)))
            
        return features
    
    # Preprocess and aggregate texture and intensity features for list of image and segmentation mask slices
    def extractFeatures (self, images, masks):
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. No features extracted.")
            return 
        
        if len(images) == 0:
            print("No images provided. No features extracted.")
            return 
        
        # Get preprocessed images and masks
        resampledImages, filteredImages, filterNames, resampledMasks = self.preprocessMultiple(images, masks)
        
        # Create feature dictionary
        features = {}
        
        # Generate aggregated shape, intensity, and texture features for original images and merge to main dictionary
        features = features | FeatureExtractor.aggregate2D(resampledImages, resampledMasks, 
                                                           self.getShapeFeatures)
        features = features | FeatureExtractor.aggregate3D(resampledImages, resampledMasks,
                                                           self.getIntensityFeatures)
        features = features | FeatureExtractor.aggregate2D(resampledImages, resampledMasks, 
                                                           self.getTextureFeatures,
                                                           discretize = bool(self.args["discretize"]))
        
        # Rename all features names as referring to original image
        features = {"original_" + k.removeprefix("original_") : v for k, v in features.items()}

        # Generate texture features for filtered images, filterNames list will be empty if no filter is applied
        for i in range(len(filterNames)):
            
            # Get all image slices with same filter applied
            currentFiltered = [filteredImages[j][i] for j in range(len(filteredImages))]
            
            # Uses default discretizaton unless image filter is LBP
            # LBP filtered images will be discretized by definition to some set number of gray levels
            # So I don't think we should discretize here
            discretize = bool(self.args["discretize"]) and (not filterNames[i].startswith("LBP"))
            
            # Get aggregated intensity and texture features for filtered images
            filteredFeatures = FeatureExtractor.aggregate3D(currentFiltered, resampledMasks, self.getIntensityFeatures)
            filteredFeatures = FeatureExtractor.aggregate2D(currentFiltered, resampledMasks, self.getTextureFeatures, discretize = discretize)
            
            # Rename features names according to filter applied
            filteredFeatures = {filterNames[i] + "_" + k.removeprefix("original_") : v for k, v in filteredFeatures.items()}
            
            # Merge with main feature dictionary
            features = features | filteredFeatures
            
        # Return merged dictionary
        return features

    # Resample 2-D image to new pixel spacing
    @staticmethod
    def resampleToSpacing (image, mask, spacing):
        # Initialize resampler
        resampler = sitk.ResampleImageFilter()
        
        resampler.SetOutputSpacing((spacing, spacing))
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        # Calculate new size using new spacing
        newSize = [round(size * (oldSpacing / newSpacing)) for 
                   size, newSpacing, oldSpacing in zip(image.GetSize(), resampler.GetOutputSpacing(), image.GetSpacing())]
        resampler.SetSize(newSize)
        
        # Linear Interpolation for Image
        resampler.SetInterpolator(sitk.sitkLinear)
        resampledImage = resampler.Execute(image)
         
        # Nearest Neighbor Interpolation for Mask
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampledMask = resampler.Execute(mask)
        
        return resampledImage, resampledMask
    
    # Aggregate feature extraction by averaging across 2D slices input
    @staticmethod
    def aggregate2D (images, masks, extractFunc, **kwargs):
        
        aggregatedFeatures = {}
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. No features extracted")
            return aggregatedFeatures        
        
        # Generate features per slice
        sliceFeatures = []
        for idx in range(len(images)):
            features = extractFunc(images[idx], masks[idx], **kwargs)
            if features:
                sliceFeatures.append(features)

        # Average features across all slices, will return empty dictionary if no features are calculated
        if len(sliceFeatures) > 0:
            for feat in sliceFeatures[0].keys():
                aggregatedFeatures[feat] = np.mean([float(featureDict[feat]) for featureDict in sliceFeatures])
            
        return aggregatedFeatures
    
    # Aggregate feature extraction by joining 2D slices input to 3D volume
    @staticmethod
    def aggregate3D (images, masks, extractFunc, **kwargs):
    
        aggregatedFeatures = {}
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. No features extracted")
            return aggregatedFeatures        
        
        # Create volume from list of slices
        imageVolume = sitk.JoinSeries(images)
        maskVolume = sitk.JoinSeries(masks)
        
        # Get intensity features
        aggregatedFeatures.update(extractFunc(imageVolume, maskVolume, **kwargs))
    
        return aggregatedFeatures