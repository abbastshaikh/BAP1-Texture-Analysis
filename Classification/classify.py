import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.feature_selection import SelectKBest, RFE, VarianceThreshold, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
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

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score, recall_score, RocCurveDisplay, auc, roc_curve
import scipy.stats

### Loading Data and Labels

# Path to data directory and feature extraction experiment
dataPath = r"D:\BAP1\Data"
expirementPath = r"D:/BAP1/Experiments/Python-FullImage_fullFeatureExtraction"

# Load extracted features and labels
features = pd.read_csv(os.path.join(expirementPath, "features.csv"))
labels = pd.read_excel(os.path.join(dataPath, "BAP1 data curation.xlsx"), 
                       sheet_name = "Disease laterality - Feng")[["Case", "Somatic BAP1 mutation status"]]

# Convert labels (in Yes/No format) to binary format
labels.rename(columns = {"Somatic BAP1 mutation status":"BAP1"}, inplace = True)
labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])

# Merge labels and features (will remove labels if we don't have features extracted)
features = pd.merge(features, labels, on = "Case", how = "left")

# Drop 'Case' column and any columns with null values
featuresNum = features.dropna(axis = 1, how = 'any').drop("Case", axis = 1)

# Create final features and labels dataframes
X, y = featuresNum.iloc[:,:-1], featuresNum.iloc[:,-1]

### SMOTE
smt = SMOTETomek(random_state = 42)
# smt = SMOTE(sampling_strategy = 1, random_state = 42)
X, y = smt.fit_resample(X, y)

### Remove Non-Robust Features
# erosionRobustness = pd.read_csv(r"D:\BAP1\Experiments\Python-FullImage_erosionFeatureRobustness\ICC.csv", header=None, names = ["Feature", "ICC"])
# toDrop = list(erosionRobustness.loc[erosionRobustness["ICC"] < 0.5]["Feature"])
# X = X.drop(columns = toDrop)

### Remove shape features
X = X[X.columns.drop(list(X.filter(regex='original_shape2D')))]

### Classification Pipeline
thres = VarianceThreshold()
variance_threshold = 0.01

scaler = StandardScaler()

def feature_selector (X, y, model):
    
    ### Remove highly correlated Features
    corr = X.corr(method = 'pearson')
    upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75) or any(upper_tri[column] < -0.75)]
    X = X.drop(columns = to_drop)
    
    ### Feature selection
   
    # selectedFeatures = X.columns
    
    # selectedFeatures = mrmr_classif(X = X, y = y, K = 10)
    
    # featureSelector = SequentialFeatureSelector(model, n_features_to_select = 10, direction = "forward")
    # featureSelector.fit(X, y)
    # selectedFeatures = list(X.columns[featureSelector.support_])
    
    featureSelector = SelectKBest(k = 10)
    featureSelector.fit(X, y)
    selectedFeatures = list(X.columns[featureSelector.get_support()])
    
    ## Random Forest
    # model.fit(X, y)
    # feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, 
    #                                     columns=['importance']).sort_values('importance', ascending=False)
    # selectedFeatures = list(feature_importances.head(20).index)

    return selectedFeatures
    
model = RandomForestClassifier(n_jobs=-1, 
                                n_estimators = 500,
                                criterion = 'entropy',
                                random_state = 42)
# model = MLPClassifier()
# model = BalancedRandomForestClassifier(n_jobs=-1, 
#                                 n_estimators = 500,
#                                 criterion = 'entropy',
#                                 random_state = 42)
# model = GradientBoostingClassifier(n_estimators = 500,
#                                     learning_rate = 0.1)
# model = DecisionTreeClassifier(max_depth = 5) 
# model = SVC(probability = True)
# model = LogisticRegression()
# model = LinearDiscriminantAnalysis()
# model = MultinomialNB()

### Perform Cross Validation
cv =  RepeatedStratifiedKFold(n_splits = 10) # StratifiedKFold(n_splits = 10, shuffle = True)
aucs = []
sensitivity = []
specificity = []

tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(7, 7), dpi = 300)

allSelectedFeatures = []

for idx, (train_index, test_index) in enumerate(cv.split(X, y)):
    
    print("Fold: " + str(idx + 1) + "/" + str(cv.get_n_splits()))
    
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Variance Threshold
    thres.fit(X_train)
    X_train = X_train.loc[:, thres.variances_ > variance_threshold]
    X_test = X_test.loc[:, thres.variances_ > variance_threshold]
    
    # Scale data
    X_train.loc[:] = scaler.fit_transform(X_train)
    X_test.loc[:] = scaler.transform(X_test)

    # Perform feature selection
    selectedFeatures = feature_selector(X_train, y_train, model)
    allSelectedFeatures.append(selectedFeatures)
    
    X_train_selected = X_train[selectedFeatures]
    X_test_selected = X_test[selectedFeatures]
    
    # Train and predict on test set
    model.fit(X_train_selected, y_train)
    
    # Calibrate model
    calibrated = CalibratedClassifierCV(model, cv = "prefit")
    calibrated.fit(X_train_selected, y_train)
    y_proba = calibrated.predict_proba(X_test_selected)
    
    # Get metrics
    y_pred = np.argmax(y_proba, axis = 1)
    aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
    sensitivity.append(sensitivity_score(y_test, y_pred))
    specificity.append(specificity_score(y_test, y_pred))
    
    print("AUC:", aucs[-1])
    print("Sensitivity:", sensitivity[-1])
    print("Specificity:", specificity[-1])
    print()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
    
    # ax.plot(
    #     fpr,
    #     tpr,
    #     lw=1,
    #     alpha=0.3
    #     )
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

### Evaluate Results
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Mean and confidence interval for classification metrics
aucCI = mean_confidence_interval(aucs)
print("Mean AUC:", aucCI[0])
print("95% Confidence Interval: (" + str(aucCI[1]) + ", " + str(aucCI[2]) + ")")
print()

sensitivityCI = mean_confidence_interval(sensitivity)
print("Mean Sensitivity:", sensitivityCI[0])
print("95% Confidence Interval: (" + str(sensitivityCI[1]) + ", " + str(sensitivityCI[2]) + ")")
print()

specificityCI = mean_confidence_interval(specificity)
print("Mean Specificity:", specificityCI[0])
print("95% Confidence Interval: (" + str(specificityCI[1]) + ", " + str(specificityCI[2]) + ")")
print()

# Get frequency of selected features
featureSelectionFreq = {feat : sum([features.count(feat) for features in allSelectedFeatures])
                        for feat in set(itertools.chain.from_iterable(allSelectedFeatures))}

freqSorted = sorted(((v, k) for k, v in featureSelectionFreq.items()), reverse=True)
for v, k in freqSorted:
    print(k, ":", v)

# Plot ROC Curve for Cross Validation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f, [%0.2f, %0.2f])" % (mean_auc, aucCI[1], aucCI[2]),
    lw=2,
    alpha=0.8,
)

ax.plot([0, 1], [0, 1], linestyle='dashed', color = "black")

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

tprs = np.array(tprs)
tprs_upper = [mean_confidence_interval(tprs[:, i])[2] for i in range(len(tprs[0]))]
tprs_lower = [mean_confidence_interval(tprs[:, i])[1] for i in range(len(tprs[0]))]

ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"95% Confidence Interval",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Cross Validation ROC Curve",
)
ax.axis("square")
ax.legend(loc="lower right")
plt.show()