import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

dataPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Data"
expirementPath = r"C:\Users\mathw\Desktop\MedIX REU\Project\Experiments\Python-EachROI"

# Load feature data
features = pd.read_csv(os.path.join(expirementPath, "features.csv"))

# Load Labels
labels = pd.read_excel(os.path.join(dataPath, "BAP1 data curation.xlsx"), 
                       sheet_name = "Disease laterality")[["Case", "Somatic BAP1 mutation status"]]

# Convert labels (in Yes/No format) to binary labels
labels.rename(columns = {"Somatic BAP1 mutation status":"BAP1"}, inplace = True)
labels["BAP1"] = labels["BAP1"].str.lower().replace(to_replace = ['yes', 'no'], value = [1, 0])

# Merge labels and features (will remove labels if we don't have features extracted)
features = pd.merge(features, labels, on = "Case", how = "left")

# Drop rows with null values, image identifier column
features = features[~features.isnull().any(axis=1)].drop("Case", axis = 1)

# Prepare and Normalize data for model
X_train, X_test, y_train, y_test = train_test_split(features.iloc[:,:-1], features.iloc[:,-1], test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25)

# Data Normalization 
# (First fit scaler to training data only, then transform all data)
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def train (model):
    # Model Training
    print("-----", model.__class__.__name__, "-----")
    num_features = [2, 5, 10, 20, 50, 99]
    for num in num_features:
        feature_selector = SelectKBest(k = num).fit(X_train, y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_val_selected = feature_selector.transform(X_val)
        X_test_selected = feature_selector.transform(X_test)
        
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)
        
        print(num, "Features")
        print("Features Selected:", feature_selector.get_feature_names_out())
        print("Accuracy:", sum(y_test == y_pred) / len(y_test))
        print("AUC:", roc_auc_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)
        
    plt.title("ROC Curve: " + model.__class__.__name__)
    plt.legend([str(num) + " Features" for num in num_features])
    plt.show()

train(SVC(), "SVM")
train(LogisticRegression())
train(LinearDiscriminantAnalysis())
train(DecisionTreeClassifier())
train(RandomForestClassifier())