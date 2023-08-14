"""
This script calculates robustness metrics for radiomic features using metrics 
of agreement over some image perturbation.

INPUT: Path to experiments for feature extraction on non-perturbed images
(originalPath) and on perturbed images (perturbationPath).

OUTPUT: CSV file with each row representing a radiomic feature and each column
representing a metric of agreement. File will be saved under the experiment
folder containing the perturbed features.

Metrics calculated include:
    Mean of Feature Ratios (MFR)
    Comparison MFR (CMFR)
    Standard Deviation of Feature Ratios (SDFR)
    Comparison SDFR (CSDFR)
    Concordance Correlation Coefficient (CCC) 
    Normalized Bias (nBias)
    Normalized Range of Agreement (nRoA)
    Intraclass Correlation Coefficient (ICC)
    ICC 95% Confidence Interval Lower Bound
    ICC 95% Confidence Interval Upper Bound
    Pearson Correlation Coefficient
    Pearson Correlation Coefficient P-Value

Written by Abbas Shaikh, Summer 2023
"""

import os
import pandas as pd
from metrics import MFR, CMFR, SDFR, CSDFR, CCC, nBias, nRoA, ICC, pearson

# Get paths to experiments containing original and perturbed features
originalPath = "D:\BAP1\Experiments\FeatureExtraction\FullFeatureExtraction"
perturbationPath = "D:\BAP1\Experiments\FeatureRobustness\Erosion"

# Load original and perturbed features
originalFeatures = pd.read_csv(os.path.join(originalPath, "features.csv"))
perturbedFeatures = pd.read_csv(os.path.join(perturbationPath, "features.csv"))

# Initialize dataframe to store robustness metrics
robustnessMetrics = pd.DataFrame(columns = ["Feature", "MFR", "CMFR", "SDFR", "CSDFR", "CCC", 
                                            "Bias", "nRoA", "ICC", "ICC_Lower", "ICC_Upper",
                                            "Pearson", "Pearson_pval"])

# For each feature extracted
featureNames = originalFeatures.columns[1:]
for feature in featureNames:
    
    print("Processing Feature:", feature)
    
    # Get feature values for all cases
    original = originalFeatures[feature]
    perturbed = perturbedFeatures[feature]
    
    # Drop cases where feature extraction failed
    null = original[original.isnull()].index.union(perturbed[perturbed.isnull()].index)
    original = original[original.index.difference(null)].reset_index(drop = True)
    perturbed = perturbed[perturbed.index.difference(null)].reset_index(drop = True)
    
    # Get ICC value and upper and lower bounds of confidence interval
    iccVal, iccCI = ICC(original, perturbed, iccType = "ICC3")
    iccLower = iccCI[0]
    iccUpper = iccCI[1]
    
    # Get Pearson coefficient value and p-value
    pearsonVal, pearsonPValue = pearson(original, perturbed)
    
    # Save to dataframe
    robustnessMetrics.loc[len(robustnessMetrics)] = [
        feature,
        MFR(original, perturbed),
        CMFR(original, perturbed),
        SDFR(original, perturbed),
        CSDFR(original, perturbed),
        CCC(original, perturbed),
        nBias(original, perturbed),
        nRoA(original, perturbed),
        iccVal,
        iccLower,
        iccUpper,
        pearsonVal,
        pearsonPValue
        ]
    
# Save to CSV file
robustnessMetrics.to_csv(os.path.join(perturbationPath, "robustnessMetrics.csv"), index = False)