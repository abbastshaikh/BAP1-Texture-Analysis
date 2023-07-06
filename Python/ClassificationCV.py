import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

dataPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Data"
expirementPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Experiments\Python-FullImage"

# Load feature data
features = pd.read_csv(os.path.join(expirementPath, "features.csv"))

# Load labels
labels = pd.read_excel(os.path.join(dataPath, "BAP1 data curation.xlsx"), 
                       sheet_name = "Disease laterality")[["Case", "Somatic BAP1 mutation status"]]

# Convert labels (in Yes/No format) to binary labels
labels.rename(columns = {"Somatic BAP1 mutation status":"BAP1"}, inplace = True)
labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])

# Merge labels and features (will remove labels if we don't have features extracted)
features = pd.merge(features, labels, on = "Case", how = "left")

# Drop rows with null values, case column
features = features[~features.isnull().any(axis=1)].drop("Case", axis = 1)

X, y = features.iloc[:,:-1], features.iloc[:,-1]

all_metrics = {}
n_features = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 99]


####

erosionCheck = pd.read_csv(r"C:\Users\mathw\Desktop\MedIX REU\Project\Experiments\erosionRobustFeatures.csv")
includeFeatures = erosionCheck[erosionCheck["pearson"] > 0.5]["feature"]
excludeFeatures = erosionCheck[erosionCheck["pearson"] < 0.5]["feature"]
X = X[X.columns.intersection(includeFeatures)]
n_features = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 83]

print(includeFeatures)
print(excludeFeatures)

####

# Note:
# Low sample size produces large bias in error on test dataset
# Hence, we perform a Leave-One-Out Cross Validation
def getModelMetrics (model, n_features):
    
    metrics = {"accuracy": [],
               "auc": [],
               "sensitivity": [],
               "specificity": []
               }
    
    print(model.__class__.__name__)        
    
    for n in n_features:
        
        pipeline = Pipeline(steps = [
            ('scaling', MinMaxScaler()),
            ('feature_selection', SelectKBest(k = n)),
            ('classification', model)
            ])
        
        cv = LeaveOneOut()
        y_prob = cross_val_predict(pipeline, X, y, cv = cv, method = 'predict_proba')
        y_pred = np.argmax(y_prob, axis = 1)
        
        metrics["accuracy"].append(sum(y == y_pred) / len(y))
        metrics["auc"].append(roc_auc_score(y, y_prob[:, 1]))
        
        cm = confusion_matrix(y, y_pred)
        
        metrics["sensitivity"].append(cm[0,0] / (cm[0,0] + cm[0,1]))
        metrics["specificity"].append(cm[1,1] / (cm[1,0] + cm[1,1]))
        
    all_metrics[model.__class__.__name__] = metrics
    
getModelMetrics(SVC(probability = True), n_features)
getModelMetrics(LogisticRegression(), n_features)
getModelMetrics(DecisionTreeClassifier(), n_features)
getModelMetrics(LinearDiscriminantAnalysis(), n_features)
# getModelMetrics(RandomForestClassifier(), n_features)

# Save to file
with pd.ExcelWriter(os.path.join(expirementPath, "results_erosionCheck.xlsx")) as writer:
    for metric in ["accuracy", "auc", "sensitivity", "specificity"]:
        df = pd.DataFrame(columns = n_features)
        
        for model in all_metrics.keys():
            df.loc[model] = all_metrics[model][metric]
        
        df.to_excel(writer, metric)
