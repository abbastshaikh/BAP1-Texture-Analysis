# Script to find overlap between RNoA, bias, MRF, and SDRF feature selection methods

# Import packages
import pandas as pd
import matplotlib as plt
from matplotlib import figure, pyplot
from statistics import median, mean
import numpy as np
import scipy.stats as st
from numpy import nan
import math

# Set value cutoffs
RNoA_cutoff = 9
bias_cutoff = 1.5
MRF_cutoff_1 = 1.1
MRF_cutoff_2 = 0.9
SDRF_cutoff = 0.5

# Intialize list of dropped features
dropped_MRF_features = []
dropped_SDRF_features =[]
dropped_RNoA_features = []
dropped_bias_features =[]

# Load in features
features = pd.read_csv(r"/Users/ilanadeutsch/Desktop/features.csv").iloc[:,1:]
erodedFeatures = pd.read_csv(r"/Users/ilanadeutsch/Desktop/featuresEroded.csv").iloc[:,1:]

# Set variables
results = []
all_feature_vals =[]

# Iterate through all features
for count, feature in enumerate(features):
    
    # Add all feature values (eroded and non eroded) to a list
    all_feature_vals.extend(features[str(feature)])
    all_feature_vals.extend(erodedFeatures[str(feature)])

    # Calculate bias
    bias = features[str(feature)] - erodedFeatures[str(feature)]

    # Eliminate NaN values
    all_feature_vals = [x for x in all_feature_vals if not(math.isnan(x))]
    bias = [x for x in bias if not(math.isnan(x))]

    # Find mean feature val
    mean_feature_val = mean(all_feature_vals)

     # Skip features with mean feature val = 0
    if mean_feature_val == 0:
        continue

    # Calculate norm bias
    norm_bias = abs(mean(bias) / mean_feature_val)

    # Calculate CI
    mean_bias = mean(bias)
    sd = np.std(bias)
    lower_lim = mean_bias - sd*1.96
    upper_lim = mean_bias + sd*1.96

    # Calculate normalized range of agreement
    nRoA = ((upper_lim - lower_lim) / abs(mean_feature_val)) 
    results.append([feature, nRoA, norm_bias])

    # Reset feature value list
    all_feature_vals = []

df = pd.DataFrame(results)

# Add features below cutoff values to a list
for featNum, feature in enumerate(df[0]):
    if df.iloc[featNum,1] > RNoA_cutoff:
        dropped_RNoA_features.append(feature)  
    if df.iloc[featNum,2] > bias_cutoff:
        dropped_bias_features.append(feature)

# Intialize variables
results = []

# Loop through all features
for count, feature in enumerate(features):
    
    # Calculate the MFR for each feature
    ratio = features[str(feature)]/erodedFeatures[str(feature)]
    ratio = [x for x in ratio if not(math.isnan(x))]

    # Skip features where all ratios are undefined
    if len(ratio) == 0:
        continue

    # Calculate mean feature ratio
    MFR = mean(ratio)

    # Calculate SDFR
    SDFR = math.sqrt(sum(([(x - MFR)**2 for x in ratio]))/len(ratio))

    # Append results to list
    results.append([feature, MFR, SDFR])

# Save results as df
df = pd.DataFrame(results)

# Add features below cutoff values to a list
for featNum, feature in enumerate(df[0]):
    if df.iloc[featNum,1] > MRF_cutoff_1:
        dropped_MRF_features.append(feature)  
    if df.iloc[featNum,1] < MRF_cutoff_2:
        dropped_MRF_features.append(feature)
    if df.iloc[featNum,2] > SDRF_cutoff:
        dropped_SDRF_features.append(feature)

# Display dropped features
print(f"Dropped MRF features: {dropped_MRF_features}")
print(f"Total = {len(dropped_MRF_features)}")
print("\n")
print(f"Dropped SDRF features: {dropped_SDRF_features}")
print(f"Total = {len(dropped_SDRF_features)}")
print("\n")
print(f"Dropped RNoA features: {dropped_RNoA_features}")
print(f"Total = {len(dropped_RNoA_features)}")
print("\n")
print(f"Dropped bias features: {dropped_bias_features}")
print(f"Total = {len(dropped_bias_features)}")
print("\n")

# Initialize lists
all_results = []
counted = []
info_list =[]
occurances =[]

# Add all removed feature lists together
all_results.extend(dropped_MRF_features)
all_results.extend(dropped_SDRF_features)
all_results.extend(dropped_RNoA_features)
all_results.extend(dropped_bias_features)

# Count how many times each feature appears in composite list
for element in all_results:
    if element not in counted:
        counted.append(element)
        occurances.append([element, all_results.count(element)])
        info_list.append(f"{element}: {all_results.count(element)}")

# Print total number of removed features
print(f"\nTotal features removed: {len(info_list)}")

# Export feature list
to_drop = pd.DataFrame(occurances)
to_drop= to_drop.sort_values(by = 1)
to_drop.to_excel(r"/Users/ilanadeutsch/Desktop/occurances.xlsx")
