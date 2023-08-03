import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

robustnessMetrics = pd.read_csv(r"D:\BAP1\Experiments\Python-FullImage_RotationRandomization\robustnessMetrics.csv")
robustnessMetrics.replace([np.inf, -np.inf], np.nan, inplace=True)

### MFR and SDFR
thresholdMFR = [0.8, 1.2]
thresholdSDFR = 0.2
thresholdPearson = 0.7
thresholdPValue = 0.05

filtered = robustnessMetrics[(robustnessMetrics["MFR"] > 0) & (robustnessMetrics["MFR"] < 2)]["MFR"]
plt.hist(filtered, bins = 25)
plt.title("MFR")
plt.axvline(x = 0.8, color = "red")
plt.axvline(x = 1.2, color = "red")
plt.show()

filtered = robustnessMetrics[(robustnessMetrics["SDFR"] > 0) & (robustnessMetrics["SDFR"] < 2)]["SDFR"]
plt.hist(filtered, bins = 25)
plt.title("SDFR")
plt.axvline(x = 0.3, color = "red")
plt.show()

plt.hist(robustnessMetrics["Pearson"], bins = 25)
plt.title("Pearson")
plt.axvline(x = 0.5, color = "red")
plt.show()

subset_MFR_SDFR = robustnessMetrics["Feature"][(robustnessMetrics["SDFR"] < thresholdSDFR) & 
                      (thresholdMFR[0] < robustnessMetrics["MFR"]) & (robustnessMetrics["MFR"] < thresholdMFR[1]) &
                      (robustnessMetrics["Pearson"] > thresholdPearson) &
                      (robustnessMetrics["Pearson_pval"] < thresholdPValue)
                      ]

print("MFR, SDFR, and Pearson (" + str(thresholdMFR[0]) + " < MFR < " + str(thresholdMFR[1]) + " and SDFR < " + str(thresholdSDFR) + " and Pearson > " + str(thresholdPearson) + " with p < " + str(thresholdPValue) +"):", 
      len(subset_MFR_SDFR),
      "Features")

### CMFR and CSDFR
thresholdCMFR = 0.2
thresholdCSDFR = 0.2

filtered = robustnessMetrics[(robustnessMetrics["CMFR"] > 0) & (robustnessMetrics["CMFR"] < 1)]["CMFR"]
plt.hist(filtered, bins = 25)
plt.title("CMFR")
plt.axvline(x = 0.2, color = "red")
plt.show()

filtered = robustnessMetrics[(robustnessMetrics["CSDFR"] > 0) & (robustnessMetrics["CSDFR"] < 2)]["CSDFR"]
plt.hist(filtered, bins = 25)
plt.title("CSDFR")
plt.axvline(x = 0.3, color = "red")
plt.show()

subset_CMFR_CSDFR = robustnessMetrics["Feature"][(robustnessMetrics["CSDFR"] < thresholdCSDFR) & 
                      (robustnessMetrics["CMFR"] < thresholdCMFR) &
                      (robustnessMetrics["Pearson"] > thresholdPearson) &
                      (robustnessMetrics["Pearson_pval"] < thresholdPValue)
                      ]

print("CMFR and CSDFR and Pearson (CMFR < " + str(thresholdCMFR) + " and CSDFR < " + str(thresholdCSDFR) +" and Pearson > " + str(thresholdPearson) + " with p < " + str(thresholdPValue) +"):", 
      len(subset_CMFR_CSDFR),
      "Features")

### nROA and Bias
thresholdNROA = 0.1
thresholdBias = [-0.2, 0.2]

filtered = robustnessMetrics[(robustnessMetrics["nROA"] > 0) & (robustnessMetrics["nROA"] < 1)]["nROA"]
plt.hist(filtered, bins = 25)
plt.title("nROA")
plt.axvline(x = thresholdNROA, color = "red")
plt.show()

filtered = robustnessMetrics[(robustnessMetrics["Bias"] > -1) & (robustnessMetrics["Bias"] < 1)]["Bias"]
plt.hist(filtered, bins = 25)
plt.title("Normalized Bias")
plt.axvline(x = thresholdBias[0], color = "red")
plt.axvline(x = thresholdBias[1], color = "red")
plt.show()

subset_nROA_Bias = robustnessMetrics["Feature"][(robustnessMetrics["nROA"] < thresholdNROA) & 
                      (thresholdBias[0] < robustnessMetrics["Bias"]) & (robustnessMetrics["Bias"] < thresholdBias[1])]

print("nROA and Normalized Bias (nROA < " + str(thresholdNROA) + " and " + str(thresholdBias[0]) + " < Bias < " + str(thresholdBias[1]) + "):", 
      len(subset_nROA_Bias),
      "Features")

### CCC
thresholdCCC = 0.9 

filtered = robustnessMetrics[(robustnessMetrics["CCC"] > -1) & (robustnessMetrics["CCC"] < 1)]["CCC"]
plt.hist(robustnessMetrics["CCC"], bins = 25)
plt.title("CCC")
plt.axvline(x = thresholdCCC, color = "red")
plt.show()

subset_CCC = robustnessMetrics["Feature"][robustnessMetrics["CCC"] > thresholdCCC]
print("CCC (Threshold = " + str(thresholdCCC) + "):", len(subset_CCC), "Features")

### ICC
thresholdICC = 0.9

filtered = robustnessMetrics[(robustnessMetrics["ICC"] > 0) & (robustnessMetrics["ICC"] < 1)]["ICC"]
plt.hist(robustnessMetrics["ICC"], bins = 25)
plt.title("ICC")
plt.axvline(x = thresholdICC, color = "red")
plt.show()

subset_ICC = robustnessMetrics["Feature"][robustnessMetrics["ICC"] > thresholdICC]
print("ICC (Threshold = " + str(thresholdICC) + "):", len(subset_ICC), "Features")

lowerICC = np.array([float(robustnessMetrics["ICC_CI"][i].split("[")[1].split()[0]) for i in range(len(robustnessMetrics))])
upperICC = np.array([float(robustnessMetrics["ICC_CI"][i].split("]")[0].split()[1]) for i in range(len(robustnessMetrics))])

filtered = lowerICC[(lowerICC > 0) & (lowerICC < 1)]
plt.hist(filtered, bins = 25)
plt.title("ICC Lower Bound")
plt.axvline(x = thresholdICC, color = "red")
plt.show()

subset_ICCLower = robustnessMetrics["Feature"][(lowerICC > thresholdICC)]
print("ICC Lower Bound (Threshold = " + str(thresholdICC) + "):", len(subset_ICCLower), "Features")

filtered = upperICC[(upperICC > 0) & (upperICC < 1)]
plt.hist(filtered, bins = 25)
plt.title("ICC Upper Bound")
plt.axvline(x = thresholdICC, color = "red")
plt.show()

subset_ICCUpper = robustnessMetrics["Feature"][(upperICC > thresholdICC)]
print("ICC Upper Bound (Threshold = " + str(thresholdICC) + "):", len(subset_ICCUpper), "Features")

subsets = [subset_MFR_SDFR,
           subset_CMFR_CSDFR,
           subset_nROA_Bias,
           subset_CCC,
           # subset_ICC,
           subset_ICCLower,
           # subset_ICCUpper
           ]

from functools import reduce
merged = reduce(lambda  left, right: pd.merge(left, right, on = 'Feature', how = 'inner'), subsets)["Feature"]

print("All Metrics: ", len(merged), "Features")


