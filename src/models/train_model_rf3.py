
#----------------------------------------------------------------------------------------------------------
# Import Libraries
#----------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image
from io import StringIO
import random
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler


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
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
# Multicase = float defination for sampling_strategy is not allowed.(Only allowed for binary case)

# Fit and resample the training data
X_train1, y_train1 = undersampler.fit_resample(X_train, y_train)

original_majority_samples = len(y_train[y_train == 2])
train1_majority_samples = len(y_train1[y_train1 == 2])
removed_samples = original_majority_samples - train1_majority_samples
print(f"Number of majority class (status_score 2) samples removed: {removed_samples}")

X_train = X_train1
y_train = y_train1
#----------------------------------------------------------------------------------------------------------
# Training Model: Random Forest
#----------------------------------------------------------------------------------------------------------
# Define your class weights as a dictionary
# class_weights = {0: 5, 1: 1.0, 2: 0.5, 3: 5} 
# rf_classifier = RandomForestClassifier(class_weight=class_weights, n_estimators=100, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=57)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report
class_report = classification_report(y_test, y_pred)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

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
# Model Evaluation: Random Forest
#----------------------------------------------------------------------------------------------------------
# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# ROC Curve
y_prob = rf_classifier.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
num_classes = len(np.unique(y_test))

for i in range(num_classes):  # num_classes is the number of classes (4 in your case)
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

plot_learning_curves(rf_classifier, "Learning Curves (Random Forest)", X_train, y_train, cv=5)
plt.show()
# No convergence, plateau and training model score is too high, underfitting & biased. 

# K-fold cross-validation
# Initializing Cross-Validation
K = 5  # Set the number of folds
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

scores = cross_val_score(rf_classifier, X, y, cv=kfold, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross Validation Scores: ", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print("Number of CV Scores used in Average: ", len(scores))
# model correctly predicted the target variable in approximately 84% of the cases during the cross-validation process.

# Result: Random Forest
# Precision: 0.77
# Recall: 0.84
# F1-Score: 0.79
# K-fold Mean Accuracy: 0.84
# ROC: Good shape | AUC Values: 0 =0.69, 1 =0.67, 2 =0.70, 3 = 0.76
# Learning Curve: underfitting & biased, model unable to learn over increasing datasize

#----------------------------------------------------------------------------------------------------------
# Hyperparameter_Fine_Tuning
#----------------------------------------------------------------------------------------------------------
rf_classifier1 = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100],  # Reduced from [50, 100, 200]
    'max_depth': [10, 20],      # Reduced from [10, 20, 30]
    'min_samples_split': [2, 5],  # Reduced from [2, 5, 10]
    'min_samples_leaf': [1, 2],   # Reduced from [1, 2, 4]
    'max_features': ['sqrt']  # Changed from 'auto' to 'sqrt'
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:")
print(f"n_estimators: {best_params['n_estimators']}")
print(f"max_depth: {best_params['max_depth']}")
print(f"min_samples_split: {best_params['min_samples_split']}")
print(f"min_samples_leaf: {best_params['min_samples_leaf']}")
print(f"max_features: {best_params['max_features']}")

y_pred1 = best_model.predict(X_test)

# If more computation power available, can ramp up the value range for each hyperparameter
default_classifier = RandomForestClassifier()

# Access the default hyperparameters
default_n_estimators = default_classifier.n_estimators
default_max_depth = default_classifier.max_depth
default_min_samples_split = default_classifier.min_samples_split
default_min_samples_leaf = default_classifier.min_samples_leaf
default_max_features = default_classifier.max_features

print("Default Parameters:")
print(f"n_estimators: {default_n_estimators}")
print(f"max_depth: {default_max_depth}")
print(f"min_samples_split: {default_min_samples_split}")
print(f"min_samples_leaf: {default_min_samples_leaf}")
print(f"max_features: {default_max_features}")

# n_estimators: 100>50 tree_reduced| max_depth: None>10 Might cause overfitting

# Calculate accuracy
accuracy1 = accuracy_score(y_test, y_pred1)

# Generate a classification report
class_report1 = classification_report(y_test, y_pred1)

# Create a confusion matrix
conf_matrix1 = confusion_matrix(y_test, y_pred1)

print(f"Accuracy: {accuracy1:.2f}")
print("Classification Report:")
print(class_report1)
print("Confusion Matrix:")
print(conf_matrix1)

# Get feature importances
feature_importances = best_model.feature_importances_

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
# Model Evaluation: Random Forest (Fine-Tuned)
#----------------------------------------------------------------------------------------------------------
# Calculate precision, recall, and F1-score
precision1 = precision_score(y_test, y_pred1, average='weighted')
recall1 = recall_score(y_test, y_pred1, average='weighted')
f11 = f1_score(y_test, y_pred1, average='weighted')

print(f'Precision: {precision1:.2f}')
print(f'Recall: {recall1:.2f}')
print(f'F1-Score: {f11:.2f}')

# ROC Curve
y_prob1 = best_model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
num_classes = len(np.unique(y_test))

for i in range(num_classes):  # num_classes is the number of classes (4 in your case)
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
plot_learning_curves(best_model, "Learning Curves (Random Forest)", X_train, y_train, cv=5)
plt.show()
# Better, but the magnitude still very small.  

# K-fold cross-validation
# Initializing Cross-Validation
K = 5  # Set the number of folds
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

scores = cross_val_score(best_model, X, y, cv=kfold, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross Validation Scores: ", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print("Number of CV Scores used in Average: ", len(scores))
# model correctly predicted the target variable in approximately 85% of the cases during the cross-validation process.

# Result: Random Forest
# Precision: 0.77
# Recall: 0.84
# F1-Score: 0.79
# K-fold Mean Accuracy: 0.85
# ROC: Good shape | AUC Values: 0 =0.69, 1 =0.67, 2 =0.70, 3 = 0.76
# Learning Curve: Better, but the magnitude still very small. 

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
        if count_status_2 <= 5:
            selected_indices.append(random_index)

# Create a sample DataFrame with the selected data points
sample_df = df.loc[selected_indices]
sample_df.head(30)
sample_df_test = sample_df.copy().drop(columns='status_score')
sample_df_test.head()

ynew = rf_classifier.predict(sample_df_test)
ynew1 = best_model.predict(sample_df_test)

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
