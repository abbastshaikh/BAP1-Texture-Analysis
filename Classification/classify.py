"""
This script performs classification on the BAP1 dataset using extracted 
radiomic features. 

The script first loads features from the specified path and labels (either 
somatic mutations or IHC as specified).

The classification pipeline then performs the following steps:
    1) Dataset Balancing (SMOTE/SMOTE-Tomek)
    2) Cross Validation
        a) Threshold Variance (i.e. removing features with low variance)
        b) Scale Feature Values
        c) Remove Correlated Features
        d) Select Top N Remaining Features
        e) Train and Predict using Calibrated Classifier
    3) Evaluation
        a) Mean and Confidence Interval for AUC, Sensitivity, and Specificity
        b) Report Most Selected Features across Cross Validation
        c) Save All Predicted Probabilities and Labels for ROC Analysis
        d) Plot Distribution of Predicted Probabilities for +/- Classes

Written by Abbas Shaikh, Summer 2023
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.feature_selection import SelectKBest, RFE, SequentialFeatureSelector
from sklearn.decomposition import PCA
from mrmr import mrmr_classif
from boruta import BorutaPy

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from util import loadClassificationData, confidenceInterval, removeCorrelatedFeatures

random_state = 100

### Load Data and Labels ###
experimentPath = r"D:\BAP1\Experiments\FeatureExtraction\FullFeatureExtraction"
labelsPath =  r"D:\BAP1\Data\BAP1Labels.csv"
labelType = "Mutation"

X, y = loadClassificationData(os.path.join(experimentPath, "features.csv"),
                                labelsPath, labelType = labelType)


### Balance Dataset ###
smt = SMOTETomek(random_state = random_state)
# smt = SMOTE(sampling_strategy = 1, random_state = random_state)
X, y = smt.fit_resample(X, y)


### Specify Classification Pipeline ###
nFeatures = 10
scaler = StandardScaler()

varianceThreshold = 0.01
correlationThreshold = 0.75

model = RandomForestClassifier(n_jobs=-1, n_estimators = 500, criterion = 'entropy', ccp_alpha = 1e-2, random_state = random_state)
# model = MLPClassifier()
# model = GradientBoostingClassifier(n_estimators = 500, random_state = random_state)
# model = XGBClassifier(n_jobs = -1, n_estimators = 100, random_state = random_state)
# model = DecisionTreeClassifier(max_depth = 5) 
# model = SVC(probability = True)
# model = LogisticRegression()
# model = LinearDiscriminantAnalysis()
# model = MultinomialNB()


### Perform Cross Validation ###
cv =  RepeatedStratifiedKFold(n_splits = 10, random_state = random_state)

AUCs = []
sensitivity = []
specificity = []

allSelectedFeatures = []
allProbs = []
allLabels = []

for idx, (train_index, test_index) in enumerate(cv.split(X, y)):
    
    print("Fold: " + str(idx + 1) + "/" + str(cv.get_n_splits()))
    
    ### Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    ### Variance Threshold
    toKeep = X_train.columns[X_train.var() > varianceThreshold]
    X_train_selected = X_train[toKeep]
    X_test_selected = X_test[toKeep]
    
    ### Scale Data
    scaler.fit(X_train_selected)
    X_train_selected.loc[:] = scaler.fit_transform(X_train_selected)
    X_test_selected.loc[:] = scaler.transform(X_test_selected)
    
    ### Remove Correlated Features
    X_train_selected = removeCorrelatedFeatures(X_train_selected, threshold = correlationThreshold)
    X_test_selected = X_test_selected[X_train_selected.columns]
    
    ### Final Feature Selection ###
    ## mRMR Feature Selection
    # selectedFeatures = mrmr_classif(X = X_train_selected, y = y_train, K = nFeatures)
    # X_train_selected = X_train_selected[selectedFeatures]
    # X_test_selected = X_test_selected[selectedFeatures]
    
    ## ANOVA Feature Selection
    selector = SelectKBest(k = nFeatures)
    selector.fit(X_train_selected, y_train)
    idx = selector.get_support(indices=True)
    X_train_selected = X_train_selected.iloc[:, idx]
    X_test_selected = X_test_selected.iloc[:, idx]
    
    ## Sequential Feature Selection
    # selector = SequentialFeatureSelector(model, n_features_to_select = nFeatures, direction = "forward")
    # selector.fit(X_train_selected, y_train)
    # idx = selector.get_support(indices=True)
    # X_train_selected = X_train_selected.iloc[:, idx]
    # X_test_selected = X_test_selected.iloc[:, idx]
    
    ## No Selection
    # X_train_selected = X_train
    # X_test_selected = X_test
    
    allSelectedFeatures.extend(X_train_selected.columns)
    
    ### Train and Predict w/ Calibrated Model
    calibrated = CalibratedClassifierCV(model)
    calibrated.fit(X_train_selected, y_train)
    y_proba = calibrated.predict_proba(X_test_selected)
    
    ### Get Evaluation Metrics
    y_pred = np.argmax(y_proba, axis = 1)
    AUCs.append(roc_auc_score(y_test, y_proba[:, 1]))
    sensitivity.append(sensitivity_score(y_test, y_pred))
    specificity.append(specificity_score(y_test, y_pred))
    
    print("AUC:", AUCs[-1])
    print("Sensitivity:", sensitivity[-1])
    print("Specificity:", specificity[-1])
    print()
    
    ### Save all probabilities and labels
    allProbs.extend(y_proba[:, 1])
    allLabels.extend(y_test)

### Evaluate Results ###
# Mean and confidence interval for classification metrics
aucCI = confidenceInterval(AUCs)
print("Mean AUC:", np.mean(AUCs))
print("95% Confidence Interval: (" + str(aucCI[0]) + ", " + str(aucCI[1]) + ")")
print()

sensitivityCI = confidenceInterval(sensitivity)
print("Mean Sensitivity:", np.mean(sensitivity))
print("95% Confidence Interval: (" + str(sensitivityCI[0]) + ", " + str(sensitivityCI[1]) + ")")
print()

specificityCI = confidenceInterval(specificity)
print("Mean Specificity:", np.mean(specificity))
print("95% Confidence Interval: (" + str(specificityCI[0]) + ", " + str(specificityCI[1]) + ")")
print()

# Frequency of selected features
counts = dict(Counter(allSelectedFeatures))
countsSorted = sorted(((v, k) for k, v in counts.items()), reverse=True)
for v, k in countsSorted:
    print(k, ":", v)

# Save all predicted probabilities and labels for ROC analysis
allProbs = np.array(allProbs)
allLabels = np.array(allLabels)

propROCtxt = np.stack([allLabels, allProbs], axis = 1)
sortedFile = propROCtxt[propROCtxt[:,0].argsort()]

lines1 = sortedFile[ : int(len(allLabels)/2) , 1] # Truly negative cases
lines2 = sortedFile[int(len(allLabels)/2) : ,1] # Truly positive cases
with open(os.path.join(experimentPath, model.__class__.__name__ + "_classification_probabilities.txt"), 'w') as f:
    f.write('BAP1 \nLarge \n')
    for line in lines1:
        f.write(str(line))
        f.write('\n')
    f.write('* \n')
    for line in lines2:
        f.write(str(line))
        f.write('\n')
    f.write('*')

# Plot distribution of probabilities by class
fig = plt.figure(figsize = (10, 7), dpi = 300)
    
plt.legend(["BAP1 [-]", "BAP1 [+]"])
plt.xlabel("Prediction Scores")
plt.xlim([0, 1])

sns.distplot(allProbs[allLabels == 0], color="blue", label = "BAP1 [-]", bins = 20)
sns.distplot(allProbs[allLabels == 1], color="red", label = "BAP1 [+]", bins = 20)

plt.show()