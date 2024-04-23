import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current working directory
current_dir = os.getcwd()

# Specify some folder names
data_dir = 'data' 
viz_dir = os.path.join(current_dir, 'data_analysis', 'viz')

data_types = ['knnimputed-k2', 'knnimputed-k3', 'knnimputed-k4', 'knnimputed-k5', 'knnimputed-k6', 'knnimputed-k7', 'knnimputed-k8', 'knnimputed-k9', 'meanimputed', 'iterativeimputed']

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define different estimators (number of trees)
    estimators = list(range(10, 150, 5))
    # Initialize lists to store metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    # Train RFC models with different stopping times
    for est in estimators:
        # Train RFC model
        rfc = RandomForestClassifier(n_estimators=est)
        rfc.fit(X_train, y_train)
        # Predict on the test set
        y_pred = rfc.predict(X_test)
        # Compute accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        # Compute the precision score
        precision = precision_score(y_test, y_pred, average='macro')
        precision_scores.append(precision)
        # Compute the recall score
        recall = recall_score(y_test, y_pred, average='macro')
        recall_scores.append(recall)
    # Plot the accuracy scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, accuracy_scores, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy Score')
    plt.title(f'Random Forest Accuracy vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'accuracy', f'{type}_accuracy_random_forest_num_trees.png'))
    plt.close()
    # Plot the precision scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, precision_scores, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Precision Score')
    plt.title(f'Random Forest Precision vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'precision', f'{type}_precision_random_forest_num_trees.png'))
    plt.close()
    # Plot the recall scores for different estimators
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, recall_scores, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Recall Score')
    plt.title(f'Random Forest Recall vs. Number of Trees ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir,'recall', f'{type}_recall_random_forest_num_trees.png'))
    plt.close()
    # Find the highest accuracy score
    highest_accuracy = max(accuracy_scores)
    # Find the index of the highest accuracy score
    index_of_highest_accuracy = accuracy_scores.index(highest_accuracy)
    # Train RFC model
    rfc = RandomForestClassifier(n_estimators=estimators[index_of_highest_accuracy])
    rfc.fit(X_train, y_train)
    # Predict on the test set
    y_pred = rfc.predict(X_test)
    # Retrieve feature importance values
    feature_importance_rfc = rfc.feature_importances_
    # Create a DataFrame to hold feature importance values and feature names
    importance_rfc_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance_rfc
    })
    # Sort the DataFrame by importance in descending order
    importance_rfc_df.sort_values(by='Importance', ascending=False, inplace=True)
    # Print the top features
    print("\nTop features by importance according to Random Forest Classifier:")
    print(importance_rfc_df.head())
    # Plot feature importance
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_rfc_df)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Random Forest Classifier Feature Importance ({type})')
    plt.savefig(os.path.join(viz_dir,'weights_and_importance', f'{type}_rfc_feature_importance.png'))
    plt.close()
    # Save the feature importance data as a CSV file
    importance_rfc_df.to_csv(os.path.join(viz_dir, 'weights_and_importance', f'{type}_rfc_feature_importance.csv'), index=False)
    print(f"Random Forest Classifier feature importance data saved to {os.path.join(viz_dir, 'weights_and_importance', f'{type}_rfc_feature_importance.csv')}")