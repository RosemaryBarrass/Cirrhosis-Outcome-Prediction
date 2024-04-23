import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current working directory
current_dir = os.getcwd()

# Specify some folder names
data_dir = 'data' 
viz_dir = os.path.join(current_dir, 'data_analysis', 'viz')

# Specify the file name
feat_file = 'cirrhosis-CLEAN.csv'

# Construct the full paths
file_path = os.path.join(current_dir, data_dir, feat_file)

# Load the dataset
data = pd.read_csv(file_path)

# Remove the data of the patients not included in the clinical trial
data = data[data['onehot__Ascites_nan'] != 1.0]
data = data.drop(columns=['onehot__Drug_nan', 'onehot__Ascites_nan', 'onehot__Hepatomegaly_nan', 'onehot__Spiders_nan'])

# Handle any other NaN values by imputing them
imputer = KNNImputer()
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the data into features and target
X = data_imputed.drop(columns=['ordinal__Status'])
y = data_imputed['ordinal__Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different estimators (number of trees)
estimators = list(range(10, 150, 5))

# Initialize lists to store accuracy scores
accuracy_scores = []

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

# Plot the accuracy scores for different estimators
plt.figure(figsize=(10, 6))
plt.plot(estimators, accuracy_scores, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy Score')
plt.title('Random Forest Accuracy vs. Number of Trees')
plt.grid(True)
plt.savefig(os.path.join(viz_dir, f'knnimpute_random_forest_num_trees.png'))
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
plt.title('Random Forest Classifier Feature Importance')
plt.savefig(os.path.join(viz_dir, f'knnimpute_rfc_feature_importance.png'))
plt.close()

# Save the feature importance data as a CSV file
importance_rfc_df.to_csv(os.path.join(viz_dir, f'knnimpute_rfc_feature_importance.csv'), index=False)
print(f"Random Forest Classifier feature importance data saved to {os.path.join(viz_dir, f'knnimpiute_rfc_feature_importance.csv')}")