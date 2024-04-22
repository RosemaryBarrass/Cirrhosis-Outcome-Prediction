import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
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

# Handle any other NaN values by imputing them
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies if needed
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the data into features and target
X = data_imputed.drop(columns=['ordinal__Status'])
y = data_imputed['ordinal__Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
rfc = RandomForestClassifier(random_state=42)

# Step 4: Initialize and train the LinearSVC classifier
svc = LinearSVC()

# Train the models on the training data
rfc.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Predict on the testing data
y_pred_rfc = rfc.predict(X_test)
y_pred_svc = svc.predict(X_test)

# Calculate accuracy
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"Accuracy for the Random Forest Classifier on the test set: {accuracy_rfc:.2f}")
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy for the Linear SVC on the test set: {accuracy_svc:.2f}")

# Retrieve feature importance values
feature_importance_rfc = rfc.feature_importances_
feature_importance_svc = svc.coef_

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
plt.figure()
sns.barplot(x='Importance', y='Feature', data=importance_rfc_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
# Save the plot as a PNG file
plt.savefig(os.path.join(viz_dir, 'rfc_feature_importance.png'))
plt.close()

# Save the feature importance data as a CSV file
importance_rfc_df.to_csv(os.path.join(viz_dir, 'rfc_feature_importance.csv'), index=False)
print(f"Random Forest feature importance data saved to {os.path.join(viz_dir, 'rfc_feature_importance.csv')}")

n_classes = len(np.unique(y))
for i in range(n_classes):
    importance_svc_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance_svc[i]
    })
    # Sort the DataFrame by importance in descending order
    importance_svc_df.sort_values(by='Importance', ascending=False, inplace=True)
    # Print the top features
    print(f"\nTop features by importance according to Linear SVC for {i}:")
    print(importance_svc_df.head())
    # Plot feature importance
    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=importance_svc_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    # Save the plot as a PNG file
    plt.savefig(os.path.join(viz_dir, f'svc_feature_importance_{i}.png'))
    plt.close()
    # Save the feature importance data as a CSV file
    importance_svc_df.to_csv(os.path.join(viz_dir, f'svc_feature_importance_{i}.csv'), index=False)
    print(f"SVC feature importance data saved to {os.path.join(viz_dir, f'svc_feature_importance.csv_{i}')}")