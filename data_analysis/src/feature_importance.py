import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
data = data[data['cat__Ascites_nan'] != 1.0]

# Handle any other NaN values by imputing them
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies if needed
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the data into features and target
X = data_imputed.drop('target_column', axis=1)  # Replace 'target_column' with your target column name
y = data_imputed['target_column']


X = data_imputed.drop(columns=['target'])
y = data_imputed['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Retrieve feature importance values
feature_importance = model.feature_importances_

# Create a DataFrame to hold feature importance values and feature names
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance in descending order
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print the top features
print("\nTop features by importance:")
print(importance_df.head())

# Plot feature importance
plt.figure()
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
# Save the plot as a PNG file
plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
plt.close()

# Save the feature importance data as a CSV file
importance_df.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)
print(f"Feature importance data saved to {os.path.join(viz_dir, 'feature_importance.csv')}")