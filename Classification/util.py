"""
This file contains various functions used in the classification scripts 
classify.py and classifyRobust.py particularly:
    loadClassificationData: This loads and prepares the extracted radiomic
        features and labels for classification.
    confidenceInterval: Determines a confidence interval for the mean of 
        a given distribution.
    removeCorrelatedFeatures: Generates an uncorrelated feature subset using 
        the pairwise Pearson correlation coefficient between all features.

Written by Abbas Shaikh, Summer 2023
"""

import pandas as pd
import numpy as np
import scipy.stats

# Loads classification data for BAP1 task
def loadClassificationData (featuresPath, labelsPath, labelType = "Mutation"):
    """
    This function loads classification data for BAP1 Mutation Status or IHC 
    status prediction.
    
    INPUT: 
        featuresPath = File to CSV containing features extracted for classification
        labelsPath = File to CSV containing classification labels
        labels = Use mutation status of IHC status as labels
        
    OUTPUT:
        X (features) and y (labels) as DataFrames
    """
    # Load extracted features
    features = pd.read_csv(featuresPath)
    
    if labelType == "Mutation":
        # Load labels CSV
        labels = pd.read_csv(labelsPath)[["Case", "Somatic BAP1 Mutation Status"]]
        labels.rename(columns = {"Somatic BAP1 Mutation Status":"BAP1"}, inplace = True)
        
        # Convert labels (in Yes/No format) to binary format
        labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])
        
    elif labelType == "IHC":
        # Load labels CSV
        labels = pd.read_csv(labelsPath)[["Case", "IHC BAP1 Status"]]
        labels.rename(columns = {"IHC BAP1 Status":"BAP1"}, inplace = True)
        
        # Drop cases where IHC status is NA or Not Done
        labels = labels.dropna(axis = 0, how = 'any')
        labels = labels[labels["BAP1"].str.lower() != "not done"]

        # Convert labels (retained/loss) to binary format
        labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['retained', 'loss', 'lost'], value = [0, 1, 1])
        
    else:
        print("Invalid label selection")
        return
        
    # Merge labels and features (remove cases if we don't have labels
    features = pd.merge(features, labels, on = "Case", how = "right")
    
    # Drop 'Case' column and any features with null values
    featuresNum = features.dropna(axis = 1, how = 'any').drop("Case", axis = 1)

    # Drop shape features (not used in classification)
    featuresNum = featuresNum[featuresNum.columns.drop(list(featuresNum.filter(regex='original_shape2D')))]

    # Create final features and labels dataframes
    X, y = featuresNum.iloc[:,:-1], featuresNum.iloc[:,-1]
    
    return X, y

# Removes features with correlation > threshold from feature set
def removeCorrelatedFeatures (features, threshold = 0.75):
    
    # Get upper triangle portion of correlation amtrix
    corrMatrix = features.corr(method = 'pearson')
    upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape),k=1).astype(bool))

    toDrop = []
    for i in range(len(upperTri.columns)):
        
        # If correlation > threshold for a feature
        if any(upperTri.iloc[:, i].abs() > threshold):
            
            # Drop the row in the matrix containing the feature, to avoid
            # dropping features correlated with it later on
            upperTri.drop(index = upperTri.index[i])
            
            # Drop feature from feature set
            toDrop.append(upperTri.columns[i])
    
    uncorrelated = features.drop(columns = upperTri[toDrop].columns)
    return uncorrelated

# Confidence interval of mean
def confidenceInterval (data, confidence = 0.95):
    
    mean, se = np.mean(data), scipy.stats.sem(data)
    width = se * scipy.stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    return mean - width, mean + width