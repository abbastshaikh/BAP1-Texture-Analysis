#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:47:44 2023

@author: shenmena
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier, PassiveAggressiveClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

import warnings
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Loading data
data = pd.read_csv('/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage_isotropicVoxelSpacing-ResampledOriginalImgs/features.csv')

# Truth labels
truthVals = pd.read_csv('/home/shenmena/MPM_U-Net/BAP1 Work/truthValues.csv')
truthVals["Truth"] = truthVals["Truth"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])

# Feature values with truth
combinedDf = pd.merge(truthVals[['Case', 'Truth']], data, on = "Case", how = "left")
# Dropping columns with any NAN vals:
combinedDf = combinedDf.dropna(axis=1)

# Separate features and target variable
X = combinedDf.drop(columns=['Case', 'Truth'])
y = combinedDf['Truth']

def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier(max_depth = 5))
    models.append(ExtraTreeClassifier())
    models.append(LinearSVC())
    models.append(SVC(probability=True))
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())
    return models

def bootstrap_auc_ci(truth, pred, alpha = 0.95, n_iterations = 2000, frac = 1):
    n_size = int(len(pred) * frac)
    stats = list()
    for i in range(n_iterations):
        truth_resampled, predictions_resampled = resample(truth, pred, n_samples=n_size)    
        score = roc_auc_score(truth_resampled, predictions_resampled)
       # print(score)
        stats.append(score)
    mean = np.mean(stats)
    se = np.std(stats)
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return mean, se, lower, upper

# Function to evaluate models and features using AUC
def evaluate_model_auc(model, X_train, y_train, X_test, y_test):
    # model.fit(X_train, y_train)

    # Check if the model has predict_proba method
    # if hasattr(model, 'predict_proba'):
    #     y_pred_proba = model.predict_proba(X_test)[:, 1]
    # else:  # For models without predict_proba, use decision_function
    #     y_pred_proba = model.decision_function(X_test)
    #     y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    
    clf = CalibratedClassifierCV(model)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc, y_pred_proba

# Repeated Stratified KFold
repeatedKFolds = 1 # A nice number from, 10, 20, 50, or even 100
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

models = get_models()
model_names = [model.__class__.__name__ for model in models]
results = []
fold_true_labels = []  # To store the ground truth labels for each fold
fold_predicted_probabilities = {model_name: [] for model_name in model_names}  # To store y_pred_proba from each fold for each model
selected_features_all_folds = {model_name: [] for model_name in model_names}  # To store selected features for each model across all folds

for t in range(repeatedKFolds):
    fold_results = []
    fold_selected_features = []  # To store selected features for each model in this fold
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_true_labels.append(y_test)  # Store y_test for bootstrapping
        
        kbest_selector = SelectKBest(f_classif, k = 129) # selecting best feature using ANOVA
        X_train_kbest = kbest_selector.fit_transform(X_train, y_train)
        X_test_kbest = kbest_selector.transform(X_test)

        # Feature selection and evaluation for each model using AUC
        for model, model_name in zip(models, model_names):
            auc, y_pred_proba = evaluate_model_auc(model, X_train_kbest, y_train, X_test_kbest, y_test)
            fold_results.append((model_name, auc))
            fold_predicted_probabilities[model_name].append(y_pred_proba)  # Store y_pred_proba for all instances

            fold_selected_features = X.columns[kbest_selector.get_support()]  # Store selected features for this model
            # print(f'{model_name} - {list(fold_selected_features)}')
            
            selected_features_all_folds[model_name].append(list(fold_selected_features))  # Store selected features for each model in this fold
           # print(f'{model_name} - Fold {t + 1} - AUC: {auc:.4f}')

    results.append(fold_results)
    
# Average AUC for each model over all folds
average_results = {}
for fold_results in results:
    for model_name, auc in fold_results:
        if model_name not in average_results:
            average_results[model_name] = []
        average_results[model_name].append(auc)

print("\nAverage Results:")
for model_name, aucs in average_results.items():
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ci_lower, ci_upper = np.percentile(aucs, [2.5, 97.5])
    print(f'{model_name}: Average AUC - {avg_auc:.4f}, Std - {std_auc:.4f}, 95% CI - ({ci_lower:.4f}, {ci_upper:.4f})')

# Perform bootstrapping on predicted probabilities
print("\nBootstrapping results:")
for model_name, y_pred_probs in fold_predicted_probabilities.items():
    y_pred_probs_all_folds = np.concatenate(y_pred_probs)  # Concatenate predicted probabilities for all instances and all folds
    y_test_all_folds = np.concatenate(fold_true_labels)  # Concatenate true labels for all instances and all folds
    
    mean_auc, se_auc, ci_lower, ci_upper = bootstrap_auc_ci(y_test_all_folds, y_pred_probs_all_folds)
    
    print(f'{model_name}: Bootstrap Mean AUC - {mean_auc:.4f}, Bootstrap Std - {se_auc:.4f}, Bootstrap 95% CI - ({ci_lower:.4f}, {ci_upper:.4f})')
    
# Print selected features for each model across all folds
print("\nSelected Features for Each Model:")
for model_name, selected_features_per_fold in selected_features_all_folds.items():
    unique_features = list(set([feature for features in selected_features_per_fold for feature in features]))
    print(f'{model_name}: {", ".join(unique_features)}')



###############################################################################
# Checking to see which feature gives us best performance

aucScores = []
featName = []

for column in X:
    colSeriesObj = X[column].values
    auc = roc_auc_score(y, colSeriesObj)
    aucScores.append(auc)
    featName.append(column)
    
np.max(aucScores)
featName[np.argmax(aucScores)]
    