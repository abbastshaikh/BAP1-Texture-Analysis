"""
This script performs classification using on the BAP1 dataset using the robust 
feature subsets as defined by various robustness metrics.

The script functions similar to classify.py , except additionally takes as 
input the path to a folder containing perturbation experiments, which contain 
files detailing  robust feature subsets. The script uses the output of 
getRobustFeatures.py, which should be run prior to this script.

NOTE: The purpose of this script is to compare classification on various robust
feature sets. Hence, the feature selection process is more limited than in 
classify.py to isolate differences between feature sets.

The classification pipeline then performs the following steps:
    1) Dataset Balancing (SMOTE/SMOTE-Tomek)
    2) Cross Validation
        a) Scale Feature Values
        b) Select Top N Features
        e) Train and Predict using Classifier
    3) Evaluation
        a) Mean and Confidence Interval for AUC, Sensitivity, and Specificity
    
The evaluation results are saved to Excel files under the experiment folder 
for each perturbation experiment analyzed. Each run will be save as a new sheet 
titled according to experimentName.

Written by Abbas Shaikh, Summer 2023
"""

import os
import pandas as pd
import numpy as np

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
from sklearn.metrics import roc_auc_score

from util import loadClassificationData, confidenceInterval

### Configurations
random_state = 100
experimentName = "ANOVA-10"
metrics = ["MFR-SDFR", "CMFR-CSDFR", "nRoA-Bias", "CCC", "ICC2", "No Metrics"]

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

### Specify Classification Pipeline
nFeatures = 10
scaler = StandardScaler()

model = RandomForestClassifier(n_jobs = -1, n_estimators = 500, criterion = 'entropy', random_state = random_state)
# model = MLPClassifier()
# model = GradientBoostingClassifier(n_estimators = 500, random_state = random_state)
# model = XGBClassifier(n_jobs = -1, n_estimators = 100, random_state = random_state)
# model = DecisionTreeClassifier(max_depth = 5) 
# model = SVC(probability = True)
# model = LogisticRegression()
# model = LinearDiscriminantAnalysis()
# model = MultinomialNB()

# Iterate over all perturbation experiments
perturbationExperiments = "D:\BAP1\Experiments\FeatureRobustness"
for subdir, dirs, files in os.walk(perturbationExperiments):
    for expDir in dirs:
        
        print("Experiment:", expDir)
        results = pd.DataFrame(columns = ["Metric", "Mean AUC", "AUC CI", "Sensitivity", "Sensitivity CI", "Specificity", "Specificity CI"])
        
        ### Load Robust Feature Sets ###
        featureRobustness = pd.read_csv(os.path.join(subdir, expDir, "robustFeatures.csv"))

        ### Iterate Over All Robustness Metrics ###
        for metric in metrics:
            
            print("Testing Metric:", metric)
            
            ### Get Robust Feature Subset
            X_robust = X[X.columns.intersection(featureRobustness[metric].dropna())]
            
            ### Perform Cross Validation ###
            cv =  RepeatedStratifiedKFold(n_splits = 10, random_state = random_state)
        
            AUCs = []
            sensitivity = []
            specificity = []
        
            for idx, (train_index, test_index) in enumerate(cv.split(X_robust, y)):
                
                ### Split data
                X_train, X_test = X_robust.iloc[train_index], X_robust.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                ### Scale Data
                scaler.fit(X_train)
                X_train.loc[:] = scaler.fit_transform(X_train)
                X_test.loc[:] = scaler.transform(X_test)
                
                ### Feature Selection ###
                ## mRMR Feature Selection
                # selectedFeatures = mrmr_classif(X = X_train_selected, y = y_train, K = nFeatures)
                # X_train_selected = X_train_selected[selectedFeatures]
                # X_test_selected = X_test_selected[selectedFeatures]
                
                ## ANOVA Feature Selection
                selector = SelectKBest(k = nFeatures)
                selector.fit(X_train, y_train)
                idx = selector.get_support(indices = True)
                X_train_selected = X_train.iloc[:, idx]
                X_test_selected = X_test.iloc[:, idx]
                
                ## Sequential Feature Selection
                # selector = SequentialFeatureSelector(model, n_features_to_select = nFeatures, direction = "forward")
                # selector.fit(X_train_selected, y_train)
                # idx = selector.get_support(indices=True)
                # X_train_selected = X_train_selected.iloc[:, idx]
                # X_test_selected = X_test_selected.iloc[:, idx]
                
                ## No Selection
                # X_train_selected = X_train
                # X_test_selected = X_test
                
                ### Train and Predict w/ Model
                model.fit(X_train_selected, y_train)
                y_proba = model.predict_proba(X_test_selected)
                
                ### Get Evaluation Metrics
                y_pred = np.argmax(y_proba, axis = 1)
                AUCs.append(roc_auc_score(y_test, y_proba[:, 1]))
                sensitivity.append(sensitivity_score(y_test, y_pred))
                specificity.append(specificity_score(y_test, y_pred))
                    
            ### Save Evaluation Metrics
            results.loc[len(results)] = [metric, 
                                         np.mean(AUCs), confidenceInterval(AUCs),
                                         np.mean(sensitivity), confidenceInterval(sensitivity),
                                         np.mean(specificity), confidenceInterval(specificity)
                                         ]
            
        ### Save Results
        outPath = os.path.join(subdir, expDir, "classification.xlsx")
        if os.path.isfile(outPath):
            writer = pd.ExcelWriter(outPath, engine = "openpyxl", mode = "a")
        else:
            writer = pd.ExcelWriter(outPath, engine = "openpyxl", mode = "w")
        results.to_excel(writer, sheet_name = experimentName, index = False)
        writer.close()