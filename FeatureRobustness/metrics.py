"""
This file contains functions for computing various metrics of agreement used
to evaluate feature robustness.

Metrics include:
    Mean of Feature Ratios (MFR)
    Comparison MFR (CMFR)
    Standard Deviation of Feature Ratios (SDFR)
    Comparison SDFR (CSDFR)
    Concordance Correlation Coefficient (CCC) 
    Normalized Bias (nBias)
    Normalized Range of Agreement (nRoA)
    Intraclass Correlation Coefficient (ICC)
    Pearson Correlation Coefficient (pearson).

Written by Abbas Shaikh, Summer 2023
"""


import pandas as pd
import numpy as np
import scipy.stats
import pingouin

# Get mean of feature ratios
def MFR (A, B):
    
    # Get feature ratios
    ratio = np.array(A) / np.array(B)
    
    # Remove null and infinite values
    ratio = ratio[~ (np.isnan(ratio) | np.isinf(ratio))]
    
    # If no values remaining, return None
    if len(ratio) == 0:
        return
    
    # Otherwise return mean feature ratio
    return np.mean(ratio)

# Get comparison mean of feature ratios
def CMFR (A, B):
    
    MFR_AB = MFR(A, B)
    MFR_BA = MFR(B, A)
    
    if MFR_AB and MFR_BA:
        return min(abs(1 - MFR_AB), abs(1 - MFR_BA))
    
    else:
        return

# Get standard deviation of feature ratios
def SDFR (A, B):
    
    # Get feature ratios
    ratio = np.array(A) / np.array(B)
    
    # Remove null and infinite values
    ratio = ratio[~ (np.isnan(ratio) | np.isinf(ratio))]
    
    # If no values remaining, return None
    if len(ratio) == 0:
        return
    
    # Otherwise return standard deviation of feature ratios
    return np.std(ratio)

# Get comparison standard deviation of feature ratios
def CSDFR (A, B):
    
    SDFR_AB = SDFR(A, B)
    SDFR_BA = SDFR(B, A)
    
    if SDFR_AB and SDFR_BA:
        return min(SDFR_AB,SDFR_BA)
    
    else:
        return

# Get Concordance Correlation Coefficient (CCC) between features
def CCC (A, B):
    
    # Get covariance matrix of features
    covariance = np.cov(A, B)
    
    # Return CCC
    return (2 * covariance[0, 1]) / \
        (covariance[0, 0] + covariance[1, 1] + (np.mean(A) - np.mean(B)) ** 2)
        
# Get normalized bias of features
def nBias (A, B):
    
    # Get bias
    bias = np.mean(A - B)
    
    # Get mean feature value
    meanValue = np.mean((A, B))
    
    # Return normalized bias
    return bias / abs(meanValue)

# Get normalized range of agreement (nROA) of features
def nRoA (A, B):
    
    # Get feature differences
    diff = A - B
    
    # Get 95% confidence interval on bias
    CI = scipy.stats.t.interval(confidence=0.95, df = len(diff) - 1,
                                loc = np.mean(diff), scale = scipy.stats.sem(diff))
    
    # Get mean feature value
    meanValue = np.mean((A, B))
    
    # Return normalized bias
    return (CI[1] - CI[0]) / abs(meanValue)

# Get ICC of given type between features
def ICC (A, B, iccType = "ICC1"):
    
    # Format dataframe for Pingouin package
    data = [[i, "A", A[i]] for i in range(len(A))]
    data.extend([[i, "B", B[i]] for i in range(len(B))])
    
    data = pd.DataFrame(data, columns = ["Case", "Rater", "Value"])
    
    # Get ICC statistics
    icc = pingouin.intraclass_corr(data = data, targets = "Case", raters = "Rater", ratings = "Value")
    
    # Return ICC value
    iccVal = icc.iloc[icc.index[icc["Type"] == iccType][0]]["ICC"]
    return iccVal

def pearson (A, B):
    
    result = scipy.stats.pearsonr(A, B)
    return result.statistic, result.pvalue
    
    
    