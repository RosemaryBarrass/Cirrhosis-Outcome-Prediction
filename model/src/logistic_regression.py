import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Remove low importance features to reduce dimensionality
    X = X.drop(columns=['numerical__Alk_Phos'])
    y = data['ordinal__Status']
    kf = KFold()
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        # Scale the data before using in the logistic regression model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Define different stopping times (number of iterations)
        stopping_times = list(range(5, 100, 5))
        # Initialize lists to store metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        # Train logistic regression models with different stopping times
        for stop_time in stopping_times:
            # Train logistic regression model
            log_reg = LogisticRegression(max_iter=stop_time, penalty='l2')
            log_reg.fit(X_train_scaled, y_train)
            # Predict on the test set
            y_pred = log_reg.predict(X_test_scaled)
            # Compute accuracy score
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            # Compute the precision score
            precision = precision_score(y_test, y_pred, average='macro')
            precision_scores.append(precision)
            # Compute the recall score
            recall = recall_score(y_test, y_pred, average='macro')
            recall_scores.append(recall)
        # Plot the accuracy scores for different stopping times
        plt.figure(figsize=(10, 6))
        plt.plot(stopping_times, accuracy_scores, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Accuracy Score')
        plt.title(f'Logistic Regression Accuracy with l2 Reg vs. Number of Iterations ({type})')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'accuracy', f'NoAlkPhos_Kfold{i}_{type}_lr_accuracy_l2_stopping_times.png'))
        plt.close()
        # Plot the precision scores for different stopping times
        plt.figure(figsize=(10, 6))
        plt.plot(stopping_times, precision_scores, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Precision Score')
        plt.title(f'Logistic Regression Precision with l2 Reg vs. Number of Iterations ({type})')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'precision', f'NoAlkPhos_Kfold{i}_{type}_lr_precision_l2_stopping_times.png'))
        plt.close()
        # Plot the recall scores for different stopping times
        plt.figure(figsize=(10, 6))
        plt.plot(stopping_times, recall_scores, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Recall Score')
        plt.title(f'Logistic Regression Recall with l2 Reg vs. Number of Iterations ({type})')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'recall', f'NoAlkPhos_Kfold{i}_{type}_lr_recall_l2_stopping_times.png'))
        plt.close()
        # Find the highest accuracy score
        highest_accuracy = max(accuracy_scores)
        # Find the index of the highest accuracy score
        index_of_highest_accuracy = accuracy_scores.index(highest_accuracy)
        # Train logistic regression model
        log_reg = LogisticRegression(max_iter=stopping_times[index_of_highest_accuracy], penalty='l2')
        log_reg.fit(X_train_scaled, y_train)
        # Predict on the test set
        y_pred = log_reg.predict(X_test_scaled)
        # Retrieve feature importance values
        feature_importance_log = log_reg.coef_[0]
        # Create a DataFrame to hold feature importance values and feature names
        importance_log_df = pd.DataFrame({
            'Feature': X.columns,
            'Weight': feature_importance_log
        })
        # Sort the DataFrame by importance in descending order
        importance_log_df.sort_values(by='Weight', ascending=False, inplace=True)
        # Plot feature importance
        plt.figure(figsize=(20, 6))
        sns.barplot(x='Weight', y='Feature', data=importance_log_df)
        plt.xlabel('Feature Weight')
        plt.ylabel('Feature')
        plt.title(f'Logistic Regression with l2 Reg Feature Weights ({type})')
        plt.savefig(os.path.join(viz_dir,'weights_and_importance', f'NoAlkPhos_Kfold{i}_{type}_lr_l2_feature_weight.png'))
        plt.close()
        # Save the feature importance data as a CSV file
        importance_log_df.to_csv(os.path.join(viz_dir,'weights_and_importance', f'NoAlkPhos_Kfold{i}_{type}_lr_l2_feature_weight.csv'), index=False)