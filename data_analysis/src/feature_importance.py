import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# Create a logistic regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the models on the training data
rfc.fit(X_train, y_train)
log_reg.fit(X_train, y_train)

# Predict on the testing data
y_pred_rfc = rfc.predict(X_test)
y_pred_log = log_reg.predict(X_test)

# Calculate accuracy
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"Accuracy for the Random Forest Classifier on the test set: {accuracy_rfc:.2f}")
accuracy_svc = accuracy_score(y_test, y_pred_log)
print(f"Accuracy for the Logistic Regression Model on the test set: {accuracy_svc:.2f}")

# Retrieve feature importance values
feature_importance_rfc = rfc.feature_importances_
feature_importance_log = log_reg.coef_[0]

# Create a DataFrame to hold feature importance values and feature names
importance_rfc_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance_rfc
})
importance_log_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance_log
})

# Sort the DataFrame by importance in descending order
importance_rfc_df.sort_values(by='Importance', ascending=False, inplace=True)
importance_log_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print the top features
print("\nTop features by importance according to Random Forest Classifier:")
print(importance_rfc_df.head())
print("\nTop features by importance according to Logistic Regression:")
print(importance_log_df.head())

# Plot feature importance
plt.figure(figsize=(20, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rfc_df)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
# Save the plot as a PNG file
plt.savefig(os.path.join(viz_dir, 'rfc_feature_importance.png'))
plt.close()

# Save the feature importance data as a CSV file
importance_rfc_df.to_csv(os.path.join(viz_dir, 'rfc_feature_importance.csv'), index=False)
print(f"Random Forest feature importance data saved to {os.path.join(viz_dir, 'rfc_feature_importance.csv')}")

# Plot feature importance
plt.figure(figsize=(20, 6))
sns.barplot(x='Importance', y='Feature', data=importance_log_df)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Logistic Regression Feature Importance')
plt.savefig(os.path.join(viz_dir, f'logreg_feature_importance.png'))
plt.close()

# Save the feature importance data as a CSV file
importance_log_df.to_csv(os.path.join(viz_dir, f'logreg_feature_importance.csv'), index=False)
print(f"Logistic Regression feature importance data saved to {os.path.join(viz_dir, f'logreg_feature_importance.csv')}")