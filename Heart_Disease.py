# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:50:53 2025

@author: Amol Gaikwad
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ===================================================================
# 📌 IMPORT REQUIRED LIBRARIES
# ===================================================================

# pandas is used to load and manipulate dataset (tables, csv files etc.)
import pandas as pd

# train_test_split helps divide data into training & testing sections
from sklearn.model_selection import train_test_split

# DecisionTreeClassifier is the machine learning model
from sklearn.tree import DecisionTreeClassifier, plot_tree

# matplotlib used for plotting / visualizing the tree and curves
import matplotlib.pyplot as plt

# Evaluation metrics
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder


# ===================================================================
# 📌 LOAD DATASET
# ===================================================================

# Reading the CSV file containing telecom churn dataset with WOE values.
# Replace filename if needed or give full path
dataframe = pd.read_csv(r"C:\Users\Amol Gaikwad\Data Science\Diceses\Heart_Disease.csv")
# Print first 5 rows to check structure
print(dataframe.head())
# ===================================================================
# 📌 SEPARATE FEATURES (X) AND TARGET (y)
# ===================================================================

# X contains independent variables (all features except target column)
df_1 = dataframe.drop("Heart_Disease", axis=1)

X = df_1.drop("Customerid", axis=1)



# y contains the dependent variable / target output we want to predict
y = dataframe["Heart_Disease"]


le = LabelEncoder()

for col in X.columns:
    
    if X[col].dtype == 'object':
        
        X[col] = le.fit_transform(X[col])



# ===================================================================
# 📌 TRAIN-TEST SPLIT
# ===================================================================

# Splitting dataset into training (70%) and testing (30%)
# random_state=42 ensures reproducibility (same results every run)
# stratify=y keeps churn proportion same in both train & test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ===================================================================
# CREATE AND TRAIN DECISION TREE MODEL
# ===================================================================

# Initialize Decision Tree model
dt = DecisionTreeClassifier(
    max_depth=4,          # limits tree depth to avoid overfitting
    min_samples_leaf=10,   # minimum customers in each leaf node
    criterion='gini'        # impurity measure used
)

# Fit model on training data (learning patterns)
dt.fit(X_train,y_train)

# ===================================================================
#  MAKE PREDICTIONS
# ===================================================================

# Predict class labels (0 or 1) for test dataset
y_pred = dt.predict(X_test)

# Predict probability values (useful for ROC curve)
y_prob = dt.predict_proba(X_test)[:, 1]   # probability of class 1 (churn)


# ===================================================================
#  MODEL EVALUATION METRICS
# ===================================================================

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# ===================================================================
# PLOT DECISION TREE
# ===================================================================

plt.figure(figsize=(25, 12))   # set large image size for better readability

# Plot tree structure with feature names and labels
plot_tree(
    dt,
    feature_names=X.columns,        # column names of features
    class_names=["Not Disease", "Disease"],  # target class labels
    filled=True,                    # color-filled boxes
    rounded=True,                   # smooth box edges
    fontsize=10
)

plt.title("Decision Tree for Heart_Disease_Prediction", fontsize=16)
plt.show()


# ===================================================================
# FEATURE IMPORTANCE (Optional & Useful)
# ===================================================================

# Identifies which features impact model results most
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Features:\n", importance_df)

importance_df.to_excel(r"C:\Users\Amol Gaikwad\Data Science\Hear_Disease\DT_Important_features.xlsx", index=False)

# ===================================================================
# Creat dataframe with Customerid Actual Churn, Predicted,Probabilites
# ===================================================================

results_df = pd.DataFrame({
    "customer_id" : X_test.index,
    "Actual_Disease" : y_test.values,
    "Predicted_Disease" : y_pred,
    "Predicted_Probability": y_prob
    })
# Rank Ordering: high probability -> Rank 1

results_df["Rank"] = results_df["Predicted_Probability"].rank( method='first',ascending=False)

# Expot to excel file

results_df.to_excel(r"C:\Users\Amol Gaikwad\Data Science\Hear_Disease\DT_diceses_predication_ranked.xlsx", index=False)

print("\n Excel Exported Sucessfuly: diceses_predication_ranked.xlsx")
print(results_df.head())