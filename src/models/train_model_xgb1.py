
#----------------------------------------------------------------------------------------------------------
# Import Libraries
#----------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import random


#----------------------------------------------------------------------------------------------------------
# M: Loading Data & Train/Test Spliting
#----------------------------------------------------------------------------------------------------------

df = pd.read_csv(r'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\processed\df4_fe.csv')
df['status_score'].value_counts(normalize=True) * 100
# Imbalanced datasets can lead to biased model predictions. Inclined to 2
# Although random forest is often more robust to class imbalance, but it doesn't really produce good result in this case.
# Use class weight/ undersampling
df.columns.tolist()
selected_columns = ['seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
                   'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity', 
                   'post_ipo_debt', 'secondary_market', 'product_crowdfunding', 'round_A', 
                   'round_B', 'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']
df.drop(columns=selected_columns, inplace=True)

# Undersampling without python
# Calculate the total number of rows to remove
total_rows_to_remove = int(0.3 * len(df[df['status_score'] == 2]))

# Get the indices of rows with 'status_score' equal to 2
indices_to_remove = df[df['status_score'] == 2].index

# Randomly select rows to remove
rows_to_remove = np.random.choice(indices_to_remove, total_rows_to_remove, replace=False)

# Create a new DataFrame with rows removed
undersampled_df = df.drop(index=rows_to_remove)


X = df.drop(columns=['status_score'])
y = df['status_score']

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree for multclassification
# Check the shape of each dataset
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Undersampling
# Define the random undersampler
# undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
# Multicase = float defination for sampling_strategy is not allowed.(Only allowed for binary case)

# Fit and resample the training data
# X_train1, y_train1 = undersampler.fit_resample(X_train, y_train)

# original_majority_samples = len(y_train[y_train == 2])
# train1_majority_samples = len(y_train1[y_train1 == 2])
# removed_samples = original_majority_samples - train1_majority_samples
# print(f"Number of majority class (status_score 2) samples removed: {removed_samples}")

# X_train = X_train1
# y_train = y_train1
#----------------------------------------------------------------------------------------------------------
# Training Model: XG Boost
#----------------------------------------------------------------------------------------------------------
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Get feature importances
feature_importances = xgb_model.feature_importances_

# Create a dataframe to associate feature names with importance scores
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

#----------------------------------------------------------------------------------------------------------
# Model Evaluation: XG Boost
#----------------------------------------------------------------------------------------------------------
# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# ROC Curve
y_prob = xgb_model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
num_classes = len(np.unique(y_test))

for i in range(num_classes):  # num_classes is the number of classes (4 in this case)
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c']  # You can change the colors as needed

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# AUC: 0.5 = random, 1 = perfect classifier
# Good model should be far away from the diagonal line

# Learning Curve
def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    for x, y_train, y_test in zip(train_sizes, train_scores_mean, test_scores_mean):
        plt.annotate(f'({x:.2f}, {y_test:.2f})', (x, y_test), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'({x:.2f}, {y_train:.2f})', (x, y_train), textcoords="offset points", xytext=(0, -20), ha='center')
    
    plt.legend(loc="best")
    return plt

plot_learning_curves(xgb_model, "Learning Curves (XG Boost)", X_train, y_train, cv=5)
plt.show()

# K-fold cross-validation
# Initializing Cross-Validation
K = 5  # Set the number of folds
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross Validation Scores: ", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print("Number of CV Scores used in Average: ", len(scores))
#----------------------------------------------------------------------------------------------------------
# Hyperparameter_Fine_Tuning
#----------------------------------------------------------------------------------------------------------
# Define the parameter grid
# Create an XGBoost classifier
xgb_classifier = XGBClassifier()

# Define a grid of hyperparameter values to search
param_grid = {
    'reg_lambda': [0.01, 0.1, 1.0],  # Specify different values for lambda
    'reg_alpha': [0.01, 0.1, 1.0]  # Specify different values for alpha
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_lambda = grid_search.best_params_['reg_lambda']
best_alpha = grid_search.best_params_['reg_alpha']
xgb_classifier.fit(X_train, y_train)

y_pred1 = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred1)
print("Accuracy:", accuracy)

# Generate a classification report
class_report = classification_report(y_test, y_pred1)
print("Classification Report:\n", class_report)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred1)
print("Confusion Matrix:\n", conf_matrix)

# Get feature importances
feature_importances = xgb_classifier.feature_importances_

# Create a dataframe to associate feature names with importance scores
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

#----------------------------------------------------------------------------------------------------------
# Model Evaluation: XG Boost (Fine-Tuned)
#----------------------------------------------------------------------------------------------------------
 # Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred1, average='weighted')
recall = recall_score(y_test, y_pred1, average='weighted')
f1 = f1_score(y_test, y_pred1, average='weighted')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# ROC Curve
y_prob1 = xgb_classifier.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
num_classes = len(np.unique(y_test))

for i in range(num_classes):  # num_classes is the number of classes (4 in this case)
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c']  # You can change the colors as needed

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# AUC: 0.5 = random, 1 = perfect classifier
# Good model should be far away from the diagonal line

# Learning Curve
def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    for x, y_train, y_test in zip(train_sizes, train_scores_mean, test_scores_mean):
        plt.annotate(f'({x:.2f}, {y_test:.2f})', (x, y_test), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'({x:.2f}, {y_train:.2f})', (x, y_train), textcoords="offset points", xytext=(0, -20), ha='center')
    
    plt.legend(loc="best")
    return plt

plot_learning_curves(xgb_classifier, "Learning Curves (XG Boost)", X_train, y_train, cv=5)
plt.show()

# K-fold cross-validation
# Initializing Cross-Validation
K = 5  # Set the number of folds
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

scores = cross_val_score(xgb_classifier, X, y, cv=kfold, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross Validation Scores: ", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print("Number of CV Scores used in Average: ", len(scores))

#----------------------------------------------------------------------------------------------------------
# Using Model to predict sample data
#----------------------------------------------------------------------------------------------------------
# Select 20 data points from df with priority for 'status_score' values 0 and 3
sample_size = 100
selected_indices = []
count_status_2 = 0  # To keep track of how many status 2 data points are added

while len(selected_indices) < sample_size:
    random_index = random.randint(0, len(df) - 1)
    
    if df.loc[random_index, 'status_score'] in [0, 3] and random_index not in selected_indices:
        selected_indices.append(random_index)
    
    if df.loc[random_index, 'status_score'] == 2:
        count_status_2 += 1
        if count_status_2 <= 80:
            selected_indices.append(random_index)

# Create a sample DataFrame with the selected data points
sample_df = df.loc[selected_indices]
sample_df.head(30)
sample_df_test = sample_df.copy().drop(columns='status_score')
sample_df_test.head()

ynew = xgb_model.predict(sample_df_test)
ynew1 = xgb_classifier.predict(sample_df_test)

# Assuming ynew and ynew1 are your predicted values
predicted_values_rf = ynew
predicted_values_tuned = ynew1
ynew
ynew1

actual_values = sample_df['status_score']

accuracy_rf = accuracy_score(actual_values, predicted_values_rf)
accuracy_tuned = accuracy_score(actual_values, predicted_values_tuned)

print(f"Accuracy (Random Forest): {accuracy_rf:.2f}")
print(f"Accuracy (Tuned Random Forest): {accuracy_tuned:.2f}")


# Class Weighting Method doesn't produce better result. The fine tuned version still cannot predict acquired and closed. 
# So far, undersampling produce best result for the model, however, it is not really good at predicting the 2 status_score.
# Maybe this has to do with the undersampling method. Too much '2' is being undersampled. 

# Using python code to reduce the sample with status_score '2' result the best. 
# With no encoding, error occur: could not convert string to float: 'FRA'

