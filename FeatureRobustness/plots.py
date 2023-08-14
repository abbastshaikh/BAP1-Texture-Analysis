"""
This script generates various relevant plots to analyzing feature robustness.

NOTE: The script will adapt the plots based on how many perturbation 
experiments are specified at the beginning.

Written by Abbas Shaikh, Summer 2023
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from statistics import mean
import os
import seaborn as sns


originalFeatures = "D:\BAP1\Experiments\FeatureExtraction\FullFeatureExtraction"

perturbationExperiments = ["D:\BAP1\Experiments\FeatureRobustness\Erosion", 
                           "D:\BAP1\Experiments\FeatureRobustness\RotationSizeAdaptation",
                           "D:\BAP1\Experiments\FeatureRobustness\RotationRandomization_5000",
                           "D:\BAP1\Experiments\FeatureRobustness\RotationSizeAdaptationRandomization_5000"]


### Feature Classes vs. Robustness Metrics ###
robustFeatures = [pd.read_csv(os.path.join(exp, "robustFeatures.csv")) for exp in perturbationExperiments]

metrics = ["MFR-SDFR", "CMFR-CSDFR", "nRoA-Bias", "CCC", "ICC"]
feature_classes = ["firstorder", "glcm", "gldm", "glrlm", "glszm", "GLDZM", "ngtdm", "NGLDM", "LTE", "FDTA", "FPS"] 

fig = plt.figure(figsize = (13, 7))

for i, metric in enumerate(metrics):
    
    proportions = []
    for feature_class in feature_classes:
        
        total = sum(robustFeatures[0]["No Metrics"].dropna().str.contains(feature_class))
        
        if not total:
            proportions.append(0)
            continue
        
        avgProportion = mean([sum(robustFeatures[j][metric].dropna().str.contains(feature_class)) / total for j in range(len(robustFeatures))])
        proportions.append(avgProportion)
        
    plt.bar(np.arange(len(proportions)) + 0.17 * i, proportions , 0.17, label = metric)

xLabels = ["First Order", "GLCM", "GLDM", "GLRLM", "GLSZM", "GLDZM", "NGTDM", "NGLDM", "Laws", "Fractal", "Fourier"]
plt.xticks(np.arange(len(proportions)) + 0.17 / 2, xLabels,
            rotation = 45,
            fontsize = 15)

plt.yticks(np.arange(0, 0.45, 0.05), fontsize = 15)

plt.xlabel("Feature Class", fontsize = 20)
plt.ylabel("Proportion of Features Selected", fontsize = 20, labelpad = 15)

plt.legend(["MFR & SDFR", "CMFR & CSDFR", "nRoA & Bias", "CCC", "ICC"], 
            fontsize = 15)
plt.show()


### Image Filters vs. Robustness Metrics ###
robustFeatures = [pd.read_csv(os.path.join(exp, "robustFeatures.csv")) for exp in perturbationExperiments]

metrics = ["MFR-SDFR", "CMFR-CSDFR", "nRoA-Bias", "CCC", "ICC"]
filters = ["original", "LoG_sigma=1.0", "LoG_sigma=2.0", "bior1.1-HH", "bior1.1-HL", "bior1.1-LH", "bior1.1-LL", "LBP"]
fig = plt.figure(figsize = (13, 7))

for i, metric in enumerate(metrics):
    
    proportions = []
    for image_filter in filters:
        
        total = sum(robustFeatures[0]["No Metrics"].dropna().str.contains(image_filter))
        
        if not total:
            proportions.append(0)
            continue
        
        avgProportion = mean([sum(robustFeatures[j][metric].dropna().str.contains(image_filter)) / total for j in range(len(robustFeatures))])
        proportions.append(avgProportion)
        
    plt.bar(np.arange(len(proportions)) + 0.17 * i, proportions , 0.17, label = metric)

xLabels = ["Original", "LoG (sigma=1.0)", "LoG (sigma=2.0)", "Wavelet (HH)", "Wavelet (HL)", "Wavelet (LH)", "Wavelet (LL)", "Local Binary Pattern"]
plt.xticks(np.arange(len(proportions)) + 0.17 / 2, xLabels,
            rotation = 45,
            fontsize = 15)

plt.yticks(np.arange(0, 0.45, 0.05), fontsize = 15)

plt.xlabel("Image Filter", fontsize = 20)
plt.ylabel("Proportion of Features Selected", fontsize = 20, labelpad = 15)

plt.legend(["MFR & SDFR", "CMFR & CSDFR", "nRoA & Bias", "CCC", "ICC"], 
            fontsize = 15)
plt.show()


### Feature Selection Stability ###
plotTitles = ["Erosion", "RV Perturbation Chain", "RC Perturbation Chain", "RVC Perturbation Chain"]

for idx, exp in enumerate(perturbationExperiments):

    fig = plt.figure(figsize = (6, 6), dpi = 300)
    
    featureStability = pd.read_csv(os.path.join(exp, "featureStability.csv"))
    numFeatures = featureStability.iloc[:, 0]
    
    for column in featureStability.iloc[:, 1:].columns:
        plt.plot(numFeatures, featureStability[column], lw = 2, marker='o')
        
    plt.xlabel("Number of Features Selected", fontsize = 15)
    plt.xticks(range(0, 60, 10), fontsize = 12)
    
    plt.ylabel("Feature Selection Stability", fontsize = 15)
    matplotlib.rc('ytick', labelsize=12)
    plt.ylim([0, 1])
    
    plt.grid(visible = True)
    
    plt.title(plotTitles[idx], fontsize = 20)
    plt.show()

# Legend for feature stability plots
fig = plt.figure(dpi = 300)
ax = plt.axes()
ax.set_facecolor("white")

legend_elements = [matplotlib.lines.Line2D([0], [0], color=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), marker = 'o', lw=2, label='MFR & SDFR'),
                   matplotlib.lines.Line2D([0], [0], color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), marker = 'o', lw=2, label='CMFR & CSDFR'),
                   matplotlib.lines.Line2D([0], [0], color=(0.3333333333333333, 0.6588235294117647, 0.40784313725490196), marker = 'o', lw=2, label='nRoA & Bias'),
                   matplotlib.lines.Line2D([0], [0], color=(0.7686274509803922, 0.3058823529411765, 0.3215686274509804), marker = 'o', lw=2, label='CCC'),
                   matplotlib.lines.Line2D([0], [0], color=(0.5058823529411764, 0.4470588235294118, 0.7019607843137254), marker = 'o', lw=2, label='ICC'),
                   matplotlib.lines.Line2D([0], [0], color=(0.5764705882352941, 0.47058823529411764, 0.3764705882352941), marker = 'o', lw=2, label='All Features')
                   ]

plt.legend(handles=legend_elements, loc='center')
plt.show()


### Number of Features Selected per Metric/Perturbation
metrics = ["MFR-SDFR", "CMFR-CSDFR", "nRoA-Bias", "CCC", "ICC", "All Metrics"]
data = np.empty((len(perturbationExperiments), len(metrics)), dtype = int)

for i, exp in enumerate(perturbationExperiments):
    robust = pd.read_csv(os.path.join(exp, "robustFeatures.csv"))
    for j, metric in enumerate(metrics):
        data[i, j] = len(robust[metric].dropna())

fig, ax = plt.subplots(figsize=(13,7), dpi = 300)

ax.set_xticks([])
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.set_yticks([])
# plt.gca().collections[0].set_clim(-50, 200)
sns.set(font_scale = 2)

sns.heatmap(data, fmt = "",cmap = 'Reds_r',annot = True,linewidths=0.30,ax=ax,yticklabels=["Erosion","RV","RC","RVC"], cbar = False)

ax.set_xticklabels(["MFR & SDFR", "CMFR & CSDFR", "nRoA & Bias", "CCC", "ICC","All"], fontsize=15)
ax.set_yticklabels(["Erosion","RV","RC","RVC"], fontsize=15)
plt.xlabel('Metric of Agreement\n', fontsize=25)
plt.ylabel('Image Perturbation\n', fontsize=25)
plt.show()


### Metric vs Feature Class and Image Filters
metric = "ICC"
robustnessMetrics = pd.read_csv(os.path.join(perturbationExperiments[0], "robustnessMetrics.csv"))
robustnessMetrics.replace([np.inf, -np.inf], np.nan, inplace=True)

fig, ax = plt.subplots(figsize=(13,7), dpi = 300)
robustnessMetrics["FeatureClass"] = np.nan 
for feature_name in ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "GLDZM", "ngtdm", "NGLDM", "LTE", "FDTA", "FPS"]:
    robustnessMetrics["FeatureClass"][robustnessMetrics["Feature"].str.contains(feature_name)] = feature_name

sns.boxplot(x='FeatureClass', y = metric, width = 0.2, data=robustnessMetrics, ax = ax)
ax.set_xticklabels(["Shape", "First Order", "GLCM", "GLDM", "GLRLM", "GLSZM", 
                    "GLDZM", "NGTDM", "NGLDM", "Laws", "Fractal", "Fourier"], rotation=45, fontsize = 15)
plt.xlabel("Feature Class", fontsize = 25)
plt.ylabel("ICC", fontsize = 25)
plt.show()

robustnessMetrics["ImageFilter"] = np.nan 
for imageFilter in ["original", "LoG_sigma=1.0", "LoG_sigma=2.0", "bior1.1-HH", 
                    "bior1.1-HL", "bior1.1-LH", "bior1.1-LL", "LBP"]:
    robustnessMetrics["ImageFilter"][robustnessMetrics["Feature"].str.contains(imageFilter)] = imageFilter
    
plt.figure(figsize = (13, 7), dpi = 300)
ax = sns.boxplot(x='ImageFilter', y = metric, width = 0.2, data = robustnessMetrics)
ax.set_xticklabels(["Original", "LoG (sigma=1.0)", "LoG (sigma=2.0)", "Wavelet (HH)", 
                    "Wavelet (HL)", "Wavelet (LH)", "Wavelet (LL)", "Local Binary Pattern"], rotation=45, fontsize = 15)
plt.xlabel("Image Filter", fontsize = 25)
plt.ylabel("ICC", fontsize = 25)

plt.show()


### Plotting Image Perturbations ### 
from FeatureExtraction.perturbation import randomize_roi_contours, erode, dilate, rotate_image_mask
import SimpleITK as sitk
import random

image = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\OriginalImgs_Resampled\Img001_0056")[:, :, 0]
mask = sitk.ReadImage(r"D:\BAP1\Data\TextureAnalysisFinal\2_18000101_CT_Axial_Chest_Recon_right\Masks_Resampled\imgs_2_18000101_CT_Axial_Chest_Recon_right_corrected_contour_1.tif")
mask = sitk.RescaleIntensity(mask, outputMinimum = 0, outputMaximum = 1)
mask[0, 0] = 0

fig, ax = plt.subplots(5, 2, figsize = (10, 25), dpi = 300)

for axes in fig.axes:
    axes.axis("off")

ax[0, 0].imshow(sitk.GetArrayFromImage(image), cmap = "gray")
ax[0, 1].imshow(sitk.GetArrayFromImage(mask), cmap = "gray")

# Erosion
ax[1, 0].imshow(sitk.GetArrayFromImage(image), cmap = "gray")
ax[1, 1].imshow(sitk.GetArrayFromImage(erode(mask, 1)), cmap = "gray")

# RV Perturbation Chain
perturbedImage, perturbedMask = rotate_image_mask(image, mask, random.randrange(45, 90))
perturbedMask = dilate(perturbedMask, 2)

ax[2, 0].imshow(sitk.GetArrayFromImage(perturbedImage), cmap = "gray")
ax[2, 1].imshow(sitk.GetArrayFromImage(perturbedMask), cmap = "gray")

# RC Perturbation Chain
perturbedImage, perturbedMask = rotate_image_mask(image, mask, random.randrange(-90, 0))
perturbedMask = randomize_roi_contours(perturbedImage, perturbedMask)

ax[3, 0].imshow(sitk.GetArrayFromImage(perturbedImage), cmap = "gray")
ax[3, 1].imshow(sitk.GetArrayFromImage(perturbedMask), cmap = "gray")

# RVC Perturbation Chain
perturbedImage, perturbedMask = rotate_image_mask(image, mask, random.randrange(180, 270))
perturbedMask = erode(perturbedMask, 2)
perturbedMask = randomize_roi_contours(perturbedImage, perturbedMask)

ax[4, 0].imshow(sitk.GetArrayFromImage(perturbedImage), cmap = "gray")
ax[4, 1].imshow(sitk.GetArrayFromImage(perturbedMask), cmap = "gray")

fig.tight_layout()