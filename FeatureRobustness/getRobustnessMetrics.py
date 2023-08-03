import os
import pandas as pd
from metrics import MFR, CMFR, SDFR, CSDFR, CCC, nBias, nROA, ICC, pearson

originalFeatures = pd.read_csv(r"D:\BAP1\Experiments\Python-FullImage_fullFeatureExtraction\features.csv")
perturbedFeatures = pd.read_csv(r"D:\BAP1\Experiments\Python-FullImage_erodedFeatures\features.csv")

outPath = r"D:\BAP1\Experiments\Python-FullImage_erodedFeatures"

robustnessMetrics = pd.DataFrame(columns = ["Feature", "MFR", "CMFR", "SDFR", "CSDFR", "CCC", 
                                            "Bias", "nROA", "ICC", "ICC_CI", "Pearson", "Pearson_pval"])

featureNames = originalFeatures.columns[1:]

for feature in featureNames:
    
    print("Processing Feature:", feature)
    
    original = originalFeatures[feature]
    perturbed = perturbedFeatures[feature]
    
    iccVal, iccCI = ICC(original, perturbed, iccType = "ICC3")
    pearsonVal, pearsonPValue = pearson(original, perturbed)
    
    robustnessMetrics.loc[len(robustnessMetrics)] = [
        feature,
        MFR(original, perturbed),
        CMFR(original, perturbed),
        SDFR(original, perturbed),
        CSDFR(original, perturbed),
        CCC(original, perturbed),
        nBias(original, perturbed),
        nROA(original, perturbed),
        iccVal,
        iccCI,
        pearsonVal,
        pearsonPValue
        ]

os.makedirs(outPath, exist_ok = True)
robustnessMetrics.to_csv(os.path.join(outPath, "robustnessMetrics.csv"), index = False)