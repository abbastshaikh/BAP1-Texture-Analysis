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
RNoA_cutoff = 3.47
bias_cutoff = 0.81
MRF_cutoff_1 = 1.56
MRF_cutoff_2 = 0.56
SDRF_cutoff = 2.50

# Load in features
features = pd.read_csv(r"/Users/ilanadeutsch/Desktop/features.csv").iloc[:,1:]
erodedFeatures = pd.read_csv(r"/Users/ilanadeutsch/Desktop/featuresEroded.csv").iloc[:,1:]

# Intialize list of dropped features
dropped_MRF_features = []
dropped_SDRF_features =[]
dropped_RNoA_features = []
dropped_bias_features =[]

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
results2 = []

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
    results2.append([feature, MFR, SDFR])

# Save results as df
df2 = pd.DataFrame(results2)

# Add features below cutoff values to a list
for featNum, feature in enumerate(df2[0]):
    if df2.iloc[featNum,1] > MRF_cutoff_1:
        dropped_MRF_features.append(feature)  
    if df2.iloc[featNum,1] < MRF_cutoff_2:
        dropped_MRF_features.append(feature)
    if df2.iloc[featNum,2] > SDRF_cutoff:
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
new_results = []

# Add all removed feature lists together
all_results.extend(dropped_MRF_features)
all_results.extend(dropped_SDRF_features)
all_results.extend(dropped_RNoA_features)
all_results.extend(dropped_bias_features)

# Create new list without duplicate results
[new_results.append(x) for x in all_results if x not in new_results]

# Create a dataframe with appropriate headers
df = pd.DataFrame(columns= ["Feature", "Bias", "nRoA", "MRF", "SDRF", "Bias_val", "nRoA_val", "MRF_val", "SDRF_val"])

# Add unique dropped features to first column
df["Feature"] = new_results

# Add a 1 for if the metric dropped the result, and a 0 for if it did not
for featNum, feature in enumerate(df["Feature"]):
    if feature in dropped_bias_features:
        df.iloc[featNum,1] = 1
    else:
        df.iloc[featNum,1] = 0
    if feature in dropped_RNoA_features:
        df.iloc[featNum,2] = 1
    else:
        df.iloc[featNum,2] = 0
    if feature in dropped_MRF_features:
        df.iloc[featNum,3] = 1
    else:
        df.iloc[featNum,4] = 0
    if feature in dropped_SDRF_features:
       df.iloc[featNum,4] = 1
    else:
        df.iloc[featNum,4] = 0

    # Add metric values to df
    for entry in results:
        if entry[0] == feature:
            df.iloc[featNum, 5] = entry[2]
            df.iloc[featNum, 6] = entry[1]
    for entry in results2:
        if entry[0] == feature:
            df.iloc[featNum, 7] = entry[1]
            df.iloc[featNum, 8] = entry[2]

# Export df
df.to_excel(r"/Users/ilanadeutsch/Desktop/feature_drop_occurences.xlsx", header = ["Feature", "Bias","nRoA","MFR","SDFR","Bias Vals","nRoA Vals","MFR Vals","SDFR Vals"])