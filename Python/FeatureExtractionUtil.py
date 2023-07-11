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
            
        print(self.pyRadiomicsExtractor.enabledFeatures)
        print(self.intensityExtractor.enabledFeatures)
        
    # Preprocess single SimpleITK image and probability map and return resampled image 
    # and mask and processed image after applying discretization and preprocessing filters
    def preprocess (self, image, probMap):
        
        # Threshold probabality map to get mask
        mask = sitk.BinaryThreshold(probMap, lowerThreshold = self.args["threshold"])
        
        # Match mask geometry to image geometry
        mask.SetDirection(image.GetDirection())
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
        
        # Resample to standardized pixel spacing
        if self.args["pixel_spacing"]:
            resampledImage, resampledMask = FeatureExtractor.resampleToSpacing(image, mask, self.args["pixel_spacing"])
                
        # Apply morphological operations to mask
        if self.args["mask_opening"]:
            resampledMask = sitk.BinaryMorphologicalOpening(resampledMask)
            
        # Discretize gray levels
        if self.args["discretize"]:
            # Get numpy arrays of image
            parameterMatrix = sitk.GetArrayFromImage(resampledImage)
            parameterMatrixCoordinates = np.nonzero(sitk.GetArrayFromImage(resampledMask))
            
            # Discretize gray levels
            if self.args["discretize"] == "fixedwidth":
                processedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                      parameterMatrixCoordinates,
                                                                      binWidth = self.args["bin_width"]
                                                                      )[0]
                
            elif self.args["discretize"] == "fixedcount":
                processedImage = radiomics.imageoperations.binImage(parameterMatrix,
                                                                      parameterMatrixCoordinates,
                                                                      binCount = self.args["n_bins"]
                                                                      )[0]
        
            # Convert back to SimpleITK image, and match geometry
            processedImage = sitk.GetImageFromArray(processedImage)
            processedImage.SetDirection(resampledImage.GetDirection())
            processedImage.SetSpacing(resampledImage.GetSpacing())
            processedImage.SetOrigin(resampledImage.GetOrigin())
            
        else:
            processedImage = resampledImage
            
        # Apply preprocessing image filters
        if self.args["preprocessing_filter"] == "LoG":
            processedImage = sitk.LaplacianRecursiveGaussian(processedImage, sigma = self.args["LoG_sigma"])
        
        return resampledImage, processedImage, resampledMask
    
    # Preprocess list of image and probabiltiy map slices
    def preprocessMultiple (self, images, probMaps):
        
        # Check if same number of images and masks are provided
        if len(images) != len(probMaps):
            print("Unequal number of images and masks. Preprocessing failed.")
            return
        
        # Compiled resample and preprocessed images and masks across
        resampledImages = []
        processedImages = []
        resampledMasks = []
        
        for idx in range(len(images)):
            resampledImage, processedImage, resampledMask = self.preprocess(images[idx], probMaps[idx])
            
            resampledImages.append(resampledImage)
            processedImages.append(processedImage)
            resampledMasks.append(resampledMask)
            
        return resampledImages, processedImages, resampledMasks
        
    # Gets texture features from single SimpleITK image and mask
    def getTextureFeatures (self, image, mask):
        
        features = {}
        
        # Converts SimpleITK images to arrays for some packages
        imgArray = sitk.GetArrayFromImage(image)
        maskArray = sitk.GetArrayFromImage(mask)
        
        # Extracts radiomic features via PyRadiomics and Nyxus
        if self.args["radiomic_features"]:
            if self.pyRadiomicsFeatures:
                features.update(dict(self.pyRadiomicsExtractor.computeFeatures(image, mask, "original")))
            if self.nyxusFeatures:
                ### Nyxus outputs dataframe, converts to dictionary
                n_features = self.nyxusExtractor.featurize(imgArray, maskArray).to_dict('list')
                n_features = {k: v[0] for k, v in n_features.items() if k.startswith(tuple(self.nyxusFeatures))}
                features.update(n_features)
                
        # Extracts Laws' features via PyFeats
        if self.args["laws_features"]:
            l_features, l_labels = pyfeats.lte_measures(imgArray, maskArray, l = 5)
            features.update(dict(zip(l_labels, l_features)))
            
        return features
    
    # Generates and aggregates texture features from list of image and mask slices
    def getAggregatedTextureFeatures (self, images, masks):
        
        aggregatedFeatures = {}
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. No features extracted")
            return aggregatedFeatures        
        
        # Generate texture features per slice
        sliceFeatures = []
        for idx in range(len(images)):
            featuresExtracted = self.getTextureFeatures(images[idx], masks[idx])
            if featuresExtracted:
                sliceFeatures.append(featuresExtracted)

        # Average features across all slices, will return empty dictionary if no features are calculated
        if len(sliceFeatures) > 0:
            for feat in sliceFeatures[0].keys():
                aggregatedFeatures[feat] = np.mean([float(featureDict[feat]) for featureDict in sliceFeatures])
            
        return aggregatedFeatures
    
    # Gets intensity features from single SimpleITK image and mask
    def getIntensityFeatures (self, image, mask):
        
        features = {}
        
        # Extracts intensity features via PyRadiomics
        if self.args["intensity_features"]:
            features.update(dict(self.intensityExtractor.computeFeatures(image, mask, "original")))

        return features
    
    # Generates and aggregates intensitys features from list of image and mask slices
    # By calculating features across entire volume
    def getAggregatedIntensityFeatures (self, images, masks):
        
        aggregatedFeatures = {}
        
        # Check if same number of images and masks are provided
        if len(images) != len(masks):
            print("Unequal number of images and masks. No features extracted")
            return aggregatedFeatures        
        
        # Create volume from list of slices
        imageVolume = sitk.JoinSeries(images)
        maskVolume = sitk.JoinSeries(masks)
        
        # Get intensity features
        aggregatedFeatures.update(self.getIntensityFeatures(imageVolume, maskVolume))

        return aggregatedFeatures
    
    # Preprocess and aggregate texture and intensity features for list of image and probability map slices
    def extractFeatures (self, images, probMaps):
        
        # Check if same number of images and masks are provided
        if len(images) != len(probMaps):
            print("Unequal number of images and probability maps. No features extracted")
            return 
        
        # Get preprocessed images and masks
        resampledImages, processedImages, resampledMasks = self.preprocessMultiple(images, probMaps)
        
        # Generate aggregated intensity features
        intensityFeatures = self.getAggregatedIntensityFeatures(resampledImages, resampledMasks)
        
        # Generate aggregated texture features
        textureFeatures = self.getAggregatedTextureFeatures(processedImages, resampledMasks)
        
        # Return merged dictionary
        return intensityFeatures | textureFeatures 

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
    
    # Get ROIs from an image and mask based on the feature extraction approach
    # Full Image vs. Patch ROI vs. Each Contiguous ROI
    # Will return a list of 1 or greater ROI
    # def getROIs (self, image, mask):
        
    #     if args["approach"] = "full":
    #         boundingBox = radiomics.imageoperations.checkMask(image, mask)[0]
    #         croppedImage, croppedMask = radiomics.imageoperations.cropToTumorMask(image, mask, boundingBox)
            
    #         return [croppedImage, croppedMask]
        
    #     if args.approach == "each":
            
    #         maskArray = sitk.GetArrayFromImage(mask)
            
    #         # Get contours of all ROIs
    #         contours = cv2.findContours(maskArray, 
    #                                     cv2.RETR_EXTERNAL, 
    #                                     cv2.CHAIN_APPROX_NONE)[0]
            
    #         # Get features from each ROI
    #         roiFeatures = []
    #         for contour in contours:
    #             boundingBox = [np.min(contour[:, :, 0]), 
    #                            np.max(contour[:, :, 0]), 
    #                            np.min(contour[:, :, 1]), 
    #                            np.max(contour[:, :, 1])]
                
    #             # If ROI is large enough, get features
    #             if np.count_nonzero(maskArray[boundingBox[2]:boundingBox[3], boundingBox[0]:boundingBox[1]]) >= minimum_nonzero:
    #                 croppedImage = image[boundingBox[0]:boundingBox[1], boundingBox[2]:boundingBox[3]]
    #                 croppedMask = mask[boundingBox[0]:boundingBox[1], boundingBox[2]:boundingBox[3]]
    #                 roiFeatures.append(getTextureFeatures(croppedImage, croppedMask, 
    #                                                       radiomic_features, 
    #                                                       laws_features, 
    #                                                       radiomicsExtractor))
            
    #         # Average features from all ROIs
    #         averageROIFeatures = {}
    #         if len(roiFeatures) > 0:
    #             for feat in roiFeatures[0].keys():
    #                 averageROIFeatures[feat] = np.mean([float(featureDict[feat]) for featureDict in roiFeatures])
            
    #     if args.approach == "patch":
            
    #         maskArray = sitk.GetArrayFromImage(mask)
            
    #         # Get indices of nonzero pixels in mask
    #         nonzero = np.array(np.nonzero(maskArray)).T
            
    #         # Randomize order so we don't prefer pixels closer to origin
    #         np.random.shuffle(nonzero)
           
    #         # Iterate till we find patch
    #         for point in nonzero:
                
    #             # Check if patch is in bounds
    #             if point[0] < maskArray.shape[0] - patch_size and point[1] < maskArray.shape[1] - patch_size:
                    
    #                 # Check if patch is all tumor pixels
    #                 if np.sum(maskArray[point[0]:point[0] + patch_size, point[1]:point[1] + patch_size]) == patch_size ** 2:
                        
    #                     imagePatch = image[point[1]:point[1] + patch_size, point[0]:point[0] + patch_size]
    #                     maskPatch = mask[point[1]:point[1] + patch_size, point[0]:point[0] + patch_size]

    #                     return getTextureFeatures(imagePatch, maskPatch, 
    #                                               radiomic_features, 
    #                                               laws_features, 
    #                                               radiomicsExtractor)
            
    #         # Return empty dictionary if no patch is found
    #         return {}
        
    
# # Crop image and mask to tumor region
# boundingBox = radiomics.imageoperations.checkMask(resampledImage, resampledMask)[0]
# resampledImage, resampledMask = radiomics.imageoperations.cropToTumorMask(resampledImage, resampledMask, boundingBox)
### GET ROIS
# Methods for aggregation ?