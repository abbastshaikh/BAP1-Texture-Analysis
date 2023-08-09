"""
This file contains functions that perform image perturbations that were used 
to assess feature robustness. Perturbations include rotation, erosion, dilation, 
and contour randomization.

Written by Abbas Shaikh, Summer 2023
"""

import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic
from scipy.ndimage import binary_closing
from skimage.transform import rotate
import warnings

warnings.filterwarnings(
    action = 'ignore',
    category = UserWarning,
    module = 'skimage',
    message = 'Applying `local_binary_pattern` to floating-point images may give unexpected results when small numerical differences between adjacent pixels are present. It is recommended to use this function with images of integer dtype'
)

# Erode binary mask (SimpleITK image) by specified radius
def erode (mask, radius):
    
    erodeFilter = sitk.BinaryErodeImageFilter()
    erodeFilter.SetKernelRadius(radius)
    eroded = erodeFilter.Execute(mask)
    
    return eroded

# Dilate binary mask (SimpleITK image) by specified radius
def dilate (mask, radius):
    
    dilateFilter = sitk.BinaryDilateImageFilter()
    dilateFilter.SetKernelRadius(radius)
    dilated = dilateFilter.Execute(mask)
    
    return dilated

# Rotate image and mask (SimpleITK images) by specified angle (degrees)
def rotate_image_mask (image, mask, angle):
    
    imageArray = sitk.GetArrayFromImage(image)
    maskArray = sitk.GetArrayFromImage(mask).astype(bool)
    
    rotatedImageArray = rotate(imageArray, angle, 
                     mode = 'constant', cval = imageArray.min(), 
                     preserve_range = True)
    
    rotatedMaskArray = rotate(maskArray, angle, 
                     mode = 'constant', cval = 0, 
                     preserve_range = True).astype(int)
    
    rotatedImage = sitk.GetImageFromArray(rotatedImageArray)
    rotatedImage.CopyInformation(image)
    
    rotatedMask = sitk.GetImageFromArray(rotatedMaskArray)
    rotatedMask.CopyInformation(mask)
    
    return rotatedImage, rotatedMask

### The below functions are based on code from the Medical Image Radiomics Processor
### https://github.com/oncoray/mirp/tree/master
### Zwanenburg A, Leger S, Agolli L, Pilz K, Troost EG, Richter C, Löck S. Assessing robustness of radiomic features by image perturbation. Scientific reports. 2019 Jan 24;9(1):614.

# Accepts SimpleITK images as input
def randomize_roi_contours(image, mask):
    """Use SLIC to randomise the roi based on supervoxels"""

    # Get supervoxels
    img_segments = get_supervoxels(image, mask)

    # Determine overlap of supervoxels with contour
    overlap_indices, overlap_fract, overlap_size = get_supervoxel_overlap(mask, img_segments)

    # Set the highest overlap to 1.0 to ensure selection of at least 1 supervoxel
    overlap_fract[np.argmax(overlap_fract)] = 1.0

    # Include supervoxels with 90% coverage and exclude those with less then 20% coverage
    overlap_fract[overlap_fract >= 0.90] = 1.0
    overlap_fract[overlap_fract < 0.20] = 0.0

    # Draw random numbers between 0.0 and 1.0
    random_incl = np.random.random(size=len(overlap_fract))
    
    # Select those segments where the random number is less than the overlap fraction - i.e. the fraction is the
    # probability of selecting the supervoxel
    incl_segments = overlap_indices[np.less(random_incl, overlap_fract)]
    
    # Replace randomised contour in original roi voxel space
    roi_vox = np.zeros(shape=mask.GetSize(), dtype=bool)
    roi_vox = np.reshape(np.in1d(np.ravel(img_segments), incl_segments), mask.GetSize())

    # Apply binary closing to close gaps
    roi_vox = binary_closing(input=roi_vox).astype(int)
    
    newMask = sitk.GetImageFromArray(roi_vox)
    newMask.CopyInformation(mask)

    return newMask

def get_supervoxels(image, mask):
    """Extracts supervoxels from an image"""
    
    # Get image object grid
    img_voxel_grid = sitk.GetArrayFromImage(image)
    maskArray = sitk.GetArrayFromImage(mask).astype(bool)

    # Get grey level thresholds
    g_range = np.empty((2))
    g_range[0] = np.min(img_voxel_grid[maskArray])
    g_range[1] = np.max(img_voxel_grid[maskArray])

    # Add 10% range outside of the grey level range
    exp_range = 0.1 * (g_range[1] - g_range[0])
    g_range = np.array([g_range[0] - exp_range, g_range[1] + exp_range])

    # Apply threshold
    img_voxel_grid[img_voxel_grid < g_range[0]] = g_range[0]
    img_voxel_grid[img_voxel_grid > g_range[1]] = g_range[1]

    # Slic constants - sigma
    sigma = 1.0 * np.min(image.GetSpacing())

    # Slic constants - number of segments
    # min_n_voxels = np.max([20.0, 500.0 / np.prod(image.GetSpacing())])
    # n_segments = int(np.prod(image.GetSize()) / min_n_voxels)
    n_segments = 5000

    # Convert to float with range [0.0, 1.0]
    if img_voxel_grid.dtype not in ["float", "float64"]:
        img_voxel_grid = img_voxel_grid.astype(float)

    img_voxel_grid -= g_range[0]
    img_voxel_grid *= 1.0 / (g_range[1]-g_range[0])
    
    # Create a slic segmentation of the image stack
    img_segments = slic(image=img_voxel_grid, n_segments=n_segments, sigma=sigma, spacing=image.GetSpacing(),
                        compactness=0.05, # multichannel=False, 
                        convert2lab=False, enforce_connectivity=True,
                        channel_axis=None)
    
    img_segments += 1

    # Release img_voxel_grid
    del img_voxel_grid
    del maskArray

    return img_segments

def get_supervoxel_overlap(mask, img_segments):
    """Determines overlap of supervoxels with other the region of interest"""

    maskArray = sitk.GetArrayFromImage(mask).astype(bool)

    # Check segments overlapping with the current contour
    overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, maskArray), return_counts=True)
    
    # Find super voxels with non-zero overlap with the roi
    overlap_size = overlap_size[overlap_segment_labels > 0]
    overlap_segment_labels = overlap_segment_labels[overlap_segment_labels > 0]

    # Check the actual size of the segments overlapping with the current contour
    segment_size = list(map(lambda x: np.sum([img_segments == x]), overlap_segment_labels))

    # Calculate the fraction of overlap
    overlap_frac = overlap_size / segment_size

    return overlap_segment_labels, overlap_frac, overlap_size