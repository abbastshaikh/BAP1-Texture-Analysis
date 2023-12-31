#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:47:44 2023

@author: shenmena
"""

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector, RFECV, VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek

import warnings
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# USING TUMOR SIZE AS THE ONLY FEATURE
# data = pd.read_excel('/home/shenmena/MPM_U-Net/BAP1 Work/Data/TumorSizes.xlsx')
# data['Tumor Area (square mm)'] = data['Tumor Area (square mm)'].astype(str)
# combinedDf = pd.merge(truthVals[['Case', 'Truth']],
#                       data, on="Case", how="left")
# # Separate features and target variable
# X = combinedDf.drop(columns=['Case', 'Truth', 'Tumor Area (number of pixels)'])
# y = combinedDf['Truth']

# Loading data
# data = pd.read_csv('/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage_preprocessingFilters_segmentedThorax/features.csv')
# data = pd.read_csv(
#     '/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage-preprocessingFilters-FractalFourier-FBN-32-segmented/features.csv')

# data = pd.read_csv(
#     '/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage_preprocessingFiltersFractalFourier-fixedbinsize-5HU/features.csv')

data = pd.read_csv(
     '/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage_preprocessingFiltersFractalFourier-FBN-32-segmented-potentialFinalResults/features.csv')
 

# Truth labels
truthVals = pd.read_csv(
    '/home/shenmena/MPM_U-Net/BAP1 Work/Data/truthValues.csv')
truthVals["Truth"] = truthVals["Truth"].str.lower().replace(
    to_replace=['yes', 'no'], value=[1, 0])

# Feature values with truth
combinedDf = pd.merge(truthVals[['Case', 'Truth']],
                      data, on="Case", how="left")
# Dropping columns with any NAN vals:
combinedDf = combinedDf.dropna(axis=1)

# Separate features and target variable
X = combinedDf.drop(columns=['Case', 'Truth'])
y = combinedDf['Truth']

 # SMOTE
smt = SMOTETomek(random_state = 8) # random state of 30 and 50 gets really low AUC (0.63)
# smt = SMOTE(sampling_strategy = 1, random_state = 42)
X, y = smt.fit_resample(X, y)
 
 ### Remove Non-Robust Features
 # erosionRobustness = pd.read_csv(r"D:\BAP1\Experiments\Python-FullImage_erosionFeatureRobustness\ICC.csv", header=None, names = ["Feature", "ICC"])
 # toDrop = list(erosionRobustness.loc[erosionRobustness["ICC"] < 0.5]["Feature"])
 # X = X.drop(columns = toDrop)
 
 ### Remove shape features
X = X[X.columns.drop(list(X.filter(regex='original_shape2D')))]


def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier(max_depth=5))
    models.append(ExtraTreeClassifier())
    models.append(LinearSVC())
    models.append(SVC(kernel='rbf', probability=True))
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier(n_estimators = 500, random_state=42,
                                          criterion='entropy', n_jobs=-1))
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis(solver='svd'))
    models.append(QuadraticDiscriminantAnalysis())
    return models


def bootstrap_auc_ci(truth, pred, alpha=0.95, n_iterations=2000, frac=1):
    n_size = int(len(pred) * frac)
    stats = list()
    for i in range(n_iterations):
        truth_resampled, predictions_resampled = resample(
            truth, pred, n_samples=n_size)
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
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc, y_pred_proba


# Repeated Stratified KFold
repeatedKFolds = 10  # A nice number from, 10, 20, 50, or even 100
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

variance_threshold = 0.01
selector = VarianceThreshold()
scaler = StandardScaler()

ANOVAtest = True
PCAFeats = False
Stepwise = False
RFEselection = False


models = get_models()
model_names = [model.__class__.__name__ for model in models]
results = []
fold_true_labels = []  # To store the ground truth labels for each fold
# To store y_pred_proba from each fold for each model
fold_predicted_probabilities = {model_name: [] for model_name in model_names}
# To store selected features for each model across all folds
selected_features_all_folds = {model_name: [] for model_name in model_names}

for t in range(repeatedKFolds):
    fold_results = []
    fold_selected_features = []  # To store selected features for each model in this fold
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_true_labels.append(y_test)  # Store y_test for bootstrapping
        
        # Apply variance threshold for the features selected
        selector.fit(X_train)
        X_train = X_train.loc[:, selector.variances_ > variance_threshold]
        X_test = X_test.loc[:, selector.variances_ > variance_threshold]
        
        # Scale data
        X_train.loc[:] = scaler.fit_transform(X_train)
        X_test.loc[:] = scaler.transform(X_test)
        
        # Remove highly correlated Features
        corr = X_train.corr(method = 'pearson')
        upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > 0.75)]
        X_train = X_train.drop(columns = to_drop)
        X_test = X_test.drop(columns = to_drop)
        
        # selecting best feature using ANOVA
        kbest_selector = SelectKBest(f_classif, k=10)
        X_train_kbest = kbest_selector.fit_transform(X_train, y_train)
        X_test_kbest = kbest_selector.transform(X_test)

        # Feature selection and evaluation for each model using AUC
        for model, model_name in zip(models, model_names):
            if ANOVAtest:
                auc, y_pred_proba = evaluate_model_auc(
                    model, X_train_kbest, y_train, X_test_kbest, y_test)
                fold_results.append((model_name, auc))
                fold_predicted_probabilities[model_name].append(
                    y_pred_proba)  # Store y_pred_proba for all instances

                # Store selected features for this model
                fold_selected_features = X_train.columns[kbest_selector.get_support(
                )]
                # print(f'{model_name} - {list(fold_selected_features)}')

                # Store selected features for each model in this fold
                selected_features_all_folds[model_name].append(
                    list(fold_selected_features))
                print(f'{model_name} - Fold {t + 1} - AUC: {auc:.4f}')
            elif PCAFeats:
                # I should scale my data here - scaled, getting AUCs of 0.5, nonsensical
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                pca = PCA(n_components=50).fit(X_train)
                
                X_train_kbest = pca.transform(X_train)
                X_test_kbest = pca.transform(X_test)
                auc, y_pred_proba = evaluate_model_auc(
                    model, X_train_kbest, y_train, X_test_kbest, y_test)

                fold_results.append((model_name, auc))
                fold_predicted_probabilities[model_name].append(
                    y_pred_proba)  # Store y_pred_proba for all instances

                print(f'{model_name} - Fold {t + 1} - AUC: {auc:.4f}')

            elif Stepwise:
                sfs = SequentialFeatureSelector(
                    model, n_features_to_select=10, direction='forward', scoring='roc_auc')  # , cv=kf)
                # sfs.fit(X_train,y_train)
                X_train_kbest = sfs.fit_transform(X_train, y_train)
                X_test_kbest = sfs.transform(X_test)
                auc, y_pred_proba = evaluate_model_auc(
                    model, X_train_kbest, y_train, X_test_kbest, y_test)

                fold_results.append((model_name, auc))
                fold_predicted_probabilities[model_name].append(
                    y_pred_proba)  # Store y_pred_proba for all instances
                # Store selected features for this model
                fold_selected_features = X.columns[sfs.get_support()]
                # print(f'{model_name} - {list(fold_selected_features)}')

                # Store selected features for each model in this fold
                selected_features_all_folds[model_name].append(
                    list(fold_selected_features))
                print(f'{model_name} - Fold {t + 1} - AUC: {auc:.4f}')

            elif RFEselection:
                rfe = RFECV(model, min_features_to_select=10,
                            scoring='roc_auc')  # , cv=kf)
                # sfs.fit(X_train,y_train)
                X_train_kbest = rfe.fit_transform(X_train, y_train)
                X_test_kbest = rfe.transform(X_test)
                auc, y_pred_proba = evaluate_model_auc(
                    model, X_train_kbest, y_train, X_test_kbest, y_test)

                fold_results.append((model_name, auc))
                fold_predicted_probabilities[model_name].append(
                    y_pred_proba)  # Store y_pred_proba for all instances
                # Store selected features for this model
                fold_selected_features = X.columns[sfs.get_support()]
                # print(f'{model_name} - {list(fold_selected_features)}')

                # Store selected features for each model in this fold
                selected_features_all_folds[model_name].append(
                    list(fold_selected_features))
                print(f'{model_name} - Fold {t + 1} - AUC: {auc:.4f}')

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
top_3_models = []
top_3_aucs = []

print("\nBootstrapping results:")
for model_name, y_pred_probs in fold_predicted_probabilities.items():
    # Concatenate predicted probabilities for all instances and all folds
    y_pred_probs_all_folds = np.concatenate(y_pred_probs)
    # Concatenate true labels for all instances and all folds
    y_test_all_folds = np.concatenate(fold_true_labels)

    mean_auc, se_auc, ci_lower, ci_upper = bootstrap_auc_ci(
        y_test_all_folds, y_pred_probs_all_folds)
    
    # Keep track of the top 3 models based on mean AUC
    top_3_models.append(model_name)
    top_3_aucs.append(mean_auc)

    print(f'{model_name}: Bootstrap Mean AUC - {mean_auc:.4f}, Bootstrap Std - {se_auc:.4f}, Bootstrap 95% CI - ({ci_lower:.4f}, {ci_upper:.4f})')

# Sort the top 3 models based on mean AUC in descending order
sorted_indices = np.argsort(top_3_aucs)[::-1]
top_3_models = np.array(top_3_models)[sorted_indices]
top_3_aucs = np.array(top_3_aucs)[sorted_indices]

# Get the predicted probabilities and true labels for the top 3 models
top_3_y_pred_probs_all_folds = []
top_3_y_test_all_folds = []
num_top_models = min(3, len(top_3_models))

for i in range(num_top_models):
    model_name = top_3_models[i]
    y_pred_probs = fold_predicted_probabilities[model_name]
    y_pred_probs_all_folds = np.concatenate(y_pred_probs)
    y_test_all_folds = np.concatenate(fold_true_labels)

    top_3_y_pred_probs_all_folds.append(y_pred_probs_all_folds)
    top_3_y_test_all_folds.append(y_test_all_folds)
    

# Top 3 models: GradientBoostingClassifier (GBC), ExtraTreesClassifier (ETC), RandomForestClassifier (RFC)
writeTXT = False
allProbas = top_3_y_pred_probs_all_folds[0] # 0 For GBC, 1 for ETC, and 2 for RFC
allLabels = top_3_y_test_all_folds[0]

if writeTXT:
    propROCtxt = np.stack([allLabels, allProbas], axis = 1)
    sortedFile = propROCtxt[propROCtxt[:,0].argsort()]
    
    lines1 = sortedFile[ : int(len(allLabels)/2) , 1] # Truly negative cases
    lines2 = sortedFile[int(len(allLabels)/2) : ,1] # Truly positive cases
    with open(r'/home/shenmena/MPM_U-Net/BAP1 Work/Data/Experiments/Python-FullImage_preprocessingFiltersFractalFourier-FBN-32-segmented-potentialFinalResults/propROC-RFC_v2.txt', 'w') as f:
        f.write('BAP1 \nLarge \n')
        for line in lines1:
            f.write(str(line))
            f.write('\n')
        f.write('* \n')
        for line in lines2:
            f.write(str(line))
            f.write('\n')
        f.write('*')
        
# Plotting the prediction scores 
import matplotlib as plt
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
import pylab

def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(round((x / 14.0129), 2)) # 14.0129 acquired from np.sum(density[0,:]) below
    return s

yfmt = tkr.FuncFormatter(numfmt)    # create your custom formatter function

# need lines 2 and 1 to be run above
plt.pyplot.xlabel("Gradient Boosting Classifier Prediction Scores")
# plt.pyplot.title("Tumor Area of the Representative Slice")
plt.pyplot.ylabel('Density') #"Frequency of Scores over the \n Repeated 10-fold cross validation")
plt.pyplot.xlim([0, 1])
density, bins, _ = plt.pyplot.hist([lines2, lines1], color=['r','b'], bins = 10, density=True, alpha=0.5,label=['BAP1 [+]','BAP1 [-]'])

# np.sum(density[0,:])
pylab.gca().yaxis.set_major_formatter(yfmt)
plt.pyplot.legend()

# Print selected features for each model across all folds
print("\nSelected Features for Each Model:")
for model_name, selected_features_per_fold in selected_features_all_folds.items():
    unique_features = list(
        set([feature for features in selected_features_per_fold for feature in features]))
    print(f'{model_name}: {", ".join(unique_features)}')
    print()

from collections import Counter
# Extract the frequency of each selected feature per model across all folds
selected_features_frequency = {model_name: Counter([feature for features in selected_features_per_fold for feature in features])
                               for model_name, selected_features_per_fold in selected_features_all_folds.items()}

# Print selected features frequency for each model
print("\nSelected Features Frequency for Each Model:")
for model_name, feature_counter in selected_features_frequency.items():
    print()
    print(f'{model_name}:')
    for feature, count in feature_counter.items():
        print(f'{feature}: {count}')
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

###############################################################################
# Checking to see if any of our features are normal - none were normal

for column in X:
    colSeriesObj = X[column].values
    stat, pvalue = stats.kstest(colSeriesObj, stats.norm.cdf)
    if pvalue > 0.05:
        print('Gaussian?')
    else:
        print('Not gaussian')

###############################################################################
# Setting up an ML pipeline
feature_names = X.columns

# Create a pipeline with MinMaxScaler and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold(threshold = (.8 * (1 - .8)))),
    ('classifier', RandomForestClassifier(n_estimators = 1000, random_state=42,
                                          criterion='entropy'))
    ])

# Create stratified k-fold cross-validation object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create empty lists to store feature importances and scores for each fold
feature_importance_list = []
auc_list = []

# Perform stratified k-fold cross-validation
nFold = 0
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
 
    # Get feature importances from the RandomForestClassifier
    feature_importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_list.append(feature_importances)
    
    # Acquiring top 50 features
    ind = np.argpartition(feature_importances, -20)[-20:]    
    topFeats = feature_names[ind]
    
    # Extract from our new X_train and X_test
    X_train = X_train[topFeats]
    X_test = X_test[topFeats]
    
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    
    # Get predicted probabilities on the test set
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    # 'ovr' for multi-class problems
    auc = roc_auc_score(y_test, y_prob)
    auc_list.append(auc)
    nFold += 1
    print(f'Fold {nFold + 1}')
    
# Convert the list of feature importances to a NumPy array
feature_importances = np.array(feature_importance_list)

# Calculate the mean feature importances and standard deviations across folds
mean_feature_importances = np.mean(feature_importances, axis=0)
std_feature_importances = np.std(feature_importances, axis=0)

# Create a DataFrame to store feature names, mean importances, and standard deviations
feature_importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Mean_Importance': mean_feature_importances, 'Std_Importance': std_feature_importances})

# Sort the DataFrame by mean importance in descending order
feature_importance_df = feature_importance_df.sort_values(
    by='Mean_Importance', ascending=False)

# Output the top features and mean AUC value
top_features = 5  # Change this value to get more or fewer top features
mean_auc = np.mean(auc_list)
print(f"Top {top_features} Features:")
print(feature_importance_df.head(top_features))
print(f"Mean AUC: {mean_auc:.4f}")

###############################################################################
# UMAP visualization
import umap
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

truthVals = pd.read_csv('/home/shenmena/MPM_U-Net/BAP1 Work/Data/truthValues.csv')
truthVals['Truth'] = (truthVals['Truth'].str.lower()).str.capitalize()

dataUMAP = combinedDf.iloc[:,2:]
scaledVecs = StandardScaler().fit_transform(dataUMAP.values, y = truthVals['Truth'])

# Define reduction
nNeighbor = 500
minDist = 1
nComponent = 2 

for nNeighbor in (2, 5, 10, 20, 50, 100, 200):
    for minDist in (0, 0.1, 0.5, 0.8, 0.99):
        reducer = umap.UMAP(n_neighbors=nNeighbor, min_dist = minDist, n_components=nComponent)
        
        # z normalization then data reduction
        scaledVecsReduced = reducer.fit_transform(scaledVecs)
        
        # Adding our binary label and converting to dataframe

        scaledVecsReducedDF = pd.DataFrame(scaledVecsReduced)
        scaledVecsReducedDF.insert(0, 'Truth', truthVals['Truth'])
        
        # Adding column headers
        scaledVecsReducedDF.columns = ['Truth', 'x', 'y']
        
        # plotting
        palette = {"No": "C0", "Yes": "C1"} # https://stackoverflow.com/questions/46173419/seaborn-change-color-according-to-hue-name
        sns.scatterplot(data=scaledVecsReducedDF, x="x", y="y", hue="Truth", marker='o', palette = palette, alpha = 1)
        plt.legend(title = 'BAP1 status', labels = ['Negative', 'Positive'])
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('C0')
        leg.legendHandles[1].set_color('C1')
        # plt.title(['n_neighbors = {}'.format(nNeighbor), 'min_dist = {}'.format(minDist), 'n_components = {}'.format(nComponent)])
        plt.title('# of neighbors = ' + str(nNeighbor) + ', minimum dist = ' + str(minDist) + ', # of components = ' + str(nComponent))
        plt.show()

# 3D UMAPs
for nNeighbor in (10, 20, 50, 100, 200):
    for minDist in (0, 0.1, 0.5, 0.8, 0.99):

        reducer = umap.UMAP(n_neighbors=nNeighbor, min_dist = minDist, n_components=3)
        
        # z normalization then data reduction
        scaledVecsReduced = reducer.fit_transform(scaledVecs)
        
        # Adding our binary label and converting to dataframe
        scaledVecsReducedDF = pd.DataFrame(scaledVecsReduced)
        scaledVecsReducedDF.insert(0, 'Truth', truthVals['Truth'])
        
        # Adding column headers
        scaledVecsReducedDF.columns = ['Truth', 'x', 'y', 'z']
        
        fig = plt.figure(figsize =(12,12))
        t = fig.suptitle("")
        ax = fig.add_subplot(111, projection='3d')
        
        BAP1pos = scaledVecsReducedDF[scaledVecsReducedDF['Truth'] == 'Yes']
        BAP1neg = scaledVecsReducedDF[scaledVecsReducedDF['Truth'] == 'No']
        
        ax.scatter(BAP1pos['x'], BAP1pos['y'], BAP1pos['z'], color = ['C1'])
        ax.scatter(BAP1neg['x'], BAP1neg['y'], BAP1neg['z'], color = ['C0'])
        
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        ax.set_title(['# of neighbors = ' + str(nNeighbor) + ', minimum dist = ' + str(minDist) + ', # of components = ' + str(3)])
        plt.legend(title="COVID Status", labels= ["BAP1 Positive","BAP1 Negative"])
        leg = ax.get_legend()
        leg.legendHandles[0].set_color("C1")
        leg.legendHandles[1].set_color("C0")

# for angle in range(0,360):
#     ax.view_init(30,angle)
#     plt.draw()
#     plt.pause(0.001)

# UMAP classification attempt
# AUC list
auc_list = []

# Create model
# svc = SVC(kernel='rbf', probability=True)
lgr = LogisticRegression()

# Create the pipeline - ignore, way too computationally intensive
# pipeline = Pipeline([("umap", umapClass), ("svc", svc)])
# params_random_pipeline = {
#     "umap__n_neighbors": [5, 200],
#     "umap__n_components": [2, 50],
#     "svc__C": [10 ** k for k in range(-3, 4, 2)],
# }

# Create stratified k-fold cross-validation object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create the RandomizedSearchCV object with 20 iterations and 20-fold cross-validation - - ignore, way too computationally intensive
# random_search = RandomizedSearchCV(pipeline, param_distributions=params_random_pipeline, 
#                                    scoring='roc_auc', n_iter=20, random_state=42)

reducer = umap.UMAP(n_neighbors=20, min_dist = 0.8, n_components=2)

nFold = 0
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scaling the vectors
    scaledX_train = StandardScaler().fit_transform(X_train, y = y_train)
    
    # Embedding with supervised UMAP
    reducer.fit(X_train, y_train)
    
    # z normalization then data reduction
    scaledX_test = StandardScaler().fit_transform(X_test)
    
    umap_scaledX_train = reducer.transform(scaledX_train)
    umap_scaledX_test = reducer.transform(scaledX_test)

    # Fit the pipeline to the training data
    # random_search.fit(X_train, y_train)
    
    # Get predicted probabilities on the test set
    # y_prob = random_search.predict_proba(X_test)[:, 1]
    
    # Evaluate calibrated model
    clf = CalibratedClassifierCV(lgr)
    clf.fit(umap_scaledX_train , y_train)
    y_pred_proba = clf.predict_proba(umap_scaledX_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate AUC
    # 'ovr' for multi-class problems
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_list.append(auc)
    nFold = nFold + 1
    print(f'Fold number: {nFold} - AUC - {auc}')


