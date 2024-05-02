import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    # Define numerical and categorical columns
    # 'ID' an 'N_Days' columns are not included
    numerical_columns = data[['numerical__Age', 'numerical__Bilirubin', 'numerical__Cholesterol', 'numerical__Albumin', 'numerical__Copper', 'numerical__Alk_Phos', 'numerical__SGOT', 'numerical__Tryglicerides', 'numerical__Platelets', 'numerical__Prothrombin']].columns
    onehot_columns = data[['bin__Drug','bin__Sex', 'bin__Ascites', 'bin__Hepatomegaly', 'bin__Spiders']].columns
    ordinal_columns = data[['ordinal__Status', 'ordinal__Edema', 'ordinal__Stage']].columns
    # Categorical Pipeline: ordinal encode data
    onehot_pipeline = Pipeline([ 
        ('onehot', OneHotEncoder(handle_unknown='error')),  # ordinal encode integer categorical data
    ])
    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer([
        ('numerical', 'passthrough', numerical_columns),    # pass through numerical data, untouched
        ('onehot', onehot_pipeline, onehot_columns),    # passthrough all binary data
        ('ordinal', 'passthrough', ordinal_columns)  # ordinal encode data with stages
    ])
    # Apply the preprocessor to the data
    cleaned_data = preprocessor.fit_transform(data)
    # Convert the cleaned data to a DataFrame if needed
    cleaned_data_df = pd.DataFrame(cleaned_data)
    # Assign column names to the cleaned data
    cleaned_data_df.columns = preprocessor.get_feature_names_out()
    # Split the data into features and target
    X = cleaned_data_df.drop(columns=['ordinal__ordinal__Status'])
    # Remove low importance features to reduce dimensionality
    # X = X.drop(columns=['numerical__Alk_Phos'])
    y = cleaned_data_df['ordinal__ordinal__Status']
    kf = KFold()
    # Define different stopping times (number of iterations)
    stopping_times = list(range(1, 20, 1))
    # Initialize averages
    avg_acc = []
    avg_prec = []
    avg_recall = []
    # Train logistic regression models with different stopping times
    for stop_time in stopping_times:
        # Initialize lists to store metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            # apply SMOTE
            X_balanced, y_balanced = apply_SMOTE(X_train, y_train)
            # Scale the data before using in the logistic regression model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_balanced)
            X_test_scaled = scaler.transform(X_test)
            # Train logistic regression model
            log_reg = LogisticRegression(max_iter=stop_time, penalty='l2')
            log_reg.fit(X_train_scaled, y_balanced)
            # Predict on the test set
            y_pred = log_reg.predict(X_test_scaled)
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
    # Plot the accuracy scores for different stopping times
    plt.figure(figsize=(10, 6))
    plt.plot(stopping_times, avg_acc, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy Score')
    plt.title(f'Logistic Regression Accuracy with l2 Reg vs. Number of Iterations ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, 'accuracy', f'AverageKFolds_{type}_lr_accuracy_l2_stopping_times.png'))
    plt.close()
    # Plot the precision scores for different stopping times
    plt.figure(figsize=(10, 6))
    plt.plot(stopping_times, avg_prec, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Precision Score')
    plt.title(f'Logistic Regression Precision with l2 Reg vs. Number of Iterations ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, 'precision', f'AverageKFolds_{type}_lr_precision_l2_stopping_times.png'))
    plt.close()
    # Plot the recall scores for different stopping times
    plt.figure(figsize=(10, 6))
    plt.plot(stopping_times, avg_recall, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Recall Score')
    plt.title(f'Logistic Regression Recall with l2 Reg vs. Number of Iterations ({type})')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, 'recall', f'AverageKFolds_{type}_lr_recall_l2_stopping_times.png'))
    plt.close()
    mean_accuracy = max(avg_acc)
    print(f'Best average accuracy: {mean_accuracy}')