import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from imblearn.over_sampling import SMOTE

def apply_SMOTE(X_train, y_train):
    smote = SMOTE(sampling_strategy='not majority')
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    return X_balanced, y_balanced

# Get the current working directory
current_dir = os.getcwd()

# Specify some folder names
data_dir = 'data' 
viz_dir = os.path.join(current_dir, 'data_analysis', 'viz')

data_types = ['knnimputed-k5']

for type in data_types:
    # Specify the file name
    feat_file = f'cirrhosis-{type}.csv'
    # Construct the full paths
    file_path = os.path.join(current_dir, data_dir, feat_file)
    # Load the dataset
    data = pd.read_csv(file_path)
    # Split the data into features and target
    X = data.drop(columns=['ordinal__Status'])
    y = data['ordinal__Status']
    # Split the data into training and testing sets
    kf = KFold()
    # Define different estimators (number of trees)
    estimators = list(range(10, 150, 5))
    # Initialize averages
    avg_acc = []
    avg_prec = []
    avg_recall = []
    # Train RFC models with different stopping times
    for est in estimators:
        # Initialize lists to store metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            X_balanced, y_balanced = apply_SMOTE(X_train, y_train)
            # Train RFC model
            rfc = RandomForestClassifier(n_estimators=est)
            rfc.fit(X_balanced, y_balanced)
            # Predict on the test set
            y_pred = rfc.predict(X_test)
            # Compute accuracy score
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            # Compute the precision score
            precision = precision_score(y_test, y_pred, average='weighted')
            precision_scores.append(precision)
            # Compute the recall score
            recall = recall_score(y_test, y_pred, average='weighted')
            recall_scores.append(recall)
        avg_acc.append(mean(accuracy_scores))
        avg_prec.append(mean(precision_scores))
        avg_recall.append(mean(recall_scores))
    # Plot the accuracy scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, avg_acc, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy Score')
    plt.title(f'Random Forest Accuracy vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'accuracy', f'AverageKFolds_{type}_accuracy_random_forest_num_trees.png'))
    plt.close()
    # Plot the precision scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, avg_prec, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Precision Score')
    plt.title(f'Random Forest Precision vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'precision', f'AverageKFolds_{type}_precision_random_forest_num_trees.png'))
    plt.close()
    # Plot the recall scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, avg_recall, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Recall Score')
    plt.title(f'Random Forest Recall vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'recall', f'AverageKFolds_{type}_recall_random_forest_num_trees.png'))
    plt.close()
    mean_accuracy = max(avg_acc)
    print(f'Best accuracies per fold: {mean_accuracy}')