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

dataPath = r"D:\BAP1\Data"
expirementPath = r"D:/BAP1/Experiments/Python-FullImage_correctedContours"

# Load feature data
features = pd.read_csv(os.path.join(expirementPath, "features.csv"))

# Load labels
labels = pd.read_excel(os.path.join(dataPath, "BAP1 data curation.xlsx"), 
                       sheet_name = "Disease laterality - Feng")[["Case", "Somatic BAP1 mutation status"]]

# Convert labels (in Yes/No format) to binary labels
labels.rename(columns = {"Somatic BAP1 mutation status":"BAP1"}, inplace = True)
labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])

# Merge labels and features (will remove labels if we don't have features extracted)
features = pd.merge(features, labels, on = "Case", how = "left")

# Drop rows with null values, case column
featuresNum = features[~features.isnull().any(axis=1)].drop("Case", axis = 1)

X, y = featuresNum.iloc[:,:-1], featuresNum.iloc[:,-1]

all_metrics = {}
n_features = range(1, len(X.columns))

####

# erosionCheck = pd.read_csv(r"C:\Users\mathw\Desktop\MedIX REU\Project\Experiments\erosionRobustFeatures.csv")
# includeFeatures = erosionCheck[erosionCheck["pearson"] > 0.5]
# excludeFeatures = erosionCheck[erosionCheck["pearson"] < 0.5]
# X = X[X.columns.intersection(includeFeatures)]

# print(includeFeatures)
# print(excludeFeatures)

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
        
        print(n)
        
        pipeline = Pipeline(steps = [
            ('scaling', MinMaxScaler()),
            ('feature_selection', SelectKBest(k = n)),
            ('classification', model)
            ])
        
        cv = StratifiedKFold(n_splits = 20) # LeaveOneOut() # StratifiedKFold(n_splits = 20) 
        y_prob = cross_val_predict(pipeline, X, y, cv = cv, method = 'predict_proba')
        y_pred = np.argmax(y_prob, axis = 1)
        
        metrics["accuracy"].append(sum(y == y_pred) / len(y))
        metrics["auc"].append(roc_auc_score(y, y_prob[:, 1]))
        
        cm = confusion_matrix(y, y_pred)
        
        metrics["sensitivity"].append(cm[0,0] / (cm[0,0] + cm[0,1]))
        metrics["specificity"].append(cm[1,1] / (cm[1,0] + cm[1,1]))
        
    all_metrics[model.__class__.__name__] = metrics
    
    plt.plot(n_features, all_metrics[model.__class__.__name__]["accuracy"])
    plt.plot(n_features, all_metrics[model.__class__.__name__]["auc"])
    plt.title(model.__class__.__name__)
    plt.legend(["Accuracy", "AUC"])
    plt.xlabel("Number of Features")
    plt.show()
#%%
    
# getModelMetrics(SVC(probability = True), n_features)
# getModelMetrics(LogisticRegression(), n_features)
getModelMetrics(DecisionTreeClassifier(max_depth = 5), n_features)
getModelMetrics(LinearDiscriminantAnalysis(), n_features)
# # getModelMetrics(RandomForestClassifier(), n_features)

# Save to file
with pd.ExcelWriter(os.path.join(expirementPath, "results.xlsx")) as writer:
    for metric in ["accuracy", "auc", "sensitivity", "specificity"]:
        df = pd.DataFrame(columns = n_features)
        
        for model in all_metrics.keys():
            df.loc[model] = all_metrics[model][metric]
        
        df.to_excel(writer, metric)
        
#%%
metrics = {"accuracy": [],
            "auc": [],
            "sensitivity": [],
            "specificity": []
            }

model = DecisionTreeClassifier()
print(model.__class__.__name__)     

n = 125

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

print(metrics)

fpr, tpr, threshold = roc_curve(y, y_prob[:, 1])

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()