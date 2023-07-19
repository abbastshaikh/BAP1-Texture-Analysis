import SimpleITK as sitk
import radiomics
import matplotlib.pyplot as plt

image = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\SegmentedThorax_Resampled\Thx001_0056")[:, :, 0]
mask = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\Masks_Resampled\imgs_2_18000101_CT_Axial_Chest_Recon_right_corrected_contour_1.tif")

# Adjusting background pixels for visualization
imageArray = sitk.GetArrayFromImage(image)
imageArray[imageArray < -1000] = -1000
image = sitk.GetImageFromArray(imageArray)

# Show original image and tumor segmentation
plt.imshow(sitk.GetArrayFromImage(image), cmap = "gray")
plt.title("Original Image")
plt.show()

plt.imshow(sitk.GetArrayFromImage(mask), cmap = "gray")
plt.title("Tumor Mask")
plt.show()

# Plotting images with Laplacian of Gaussian (LoG) filters
for sigma in [0.5, 1, 2, 4, 8]:
    LoGImage = sitk.LaplacianRecursiveGaussian(image, sigma = sigma)
    
    plt.imshow(sitk.GetArrayFromImage(LoGImage), cmap = "gray")
    plt.title("Laplacian of Gaussian, Sigma = " + str(sigma))
    plt.show()

# Plotting images with Local Binary Pattern (LBP) filters
for radius in [1, 2, 4]:
    LBPImage = next(radiomics.imageoperations.getLBP2DImage(image, mask, lbp2DRadius = radius))[0]
    plt.imshow(sitk.GetArrayFromImage(LBPImage), cmap = "gray")
    plt.title("Local Binary Pattern, Radius = " + str(radius))
    plt.show()

# Plotting images with wavelet filters
for wavelet in ['haar', 'dmey', 'coif1', 'sym2', 'db2', 'bior1.1', 'rbio1.1']:
    
    waveletTransform = radiomics.imageoperations.getWaveletImage(image, mask, wavelet = wavelet)
    
    # Iterating through all decompositions
    while True:
        try:
            waveletImage = next(waveletTransform)
            plt.imshow(sitk.GetArrayFromImage(waveletImage[0]), cmap = "gray")
            plt.title(wavelet + " " + waveletImage[1])
            plt.show()
        except StopIteration:
            break