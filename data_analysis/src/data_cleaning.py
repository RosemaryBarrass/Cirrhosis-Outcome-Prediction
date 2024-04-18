import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Get the current working directory
current_dir = os.getcwd()

# Specify the folder name in the directory
data_dir = 'data' 

# Specify the file name
file_name = 'cirrhosis-RAW.csv'

# Construct the full path to the file
file_path = os.path.join(current_dir, data_dir, file_name)

# Read the raw data
raw_data = pd.read_csv(file_path)
target_data = raw_data['Status']
raw_data = raw_data.drop(columns=['ID','Status'])

### Data Cleaning ###

# Check for missing values
print(raw_data.isnull().sum())

# Define numerical and categorical columns
numerical_columns = raw_data.select_dtypes(include=np.number).columns
categorical_columns = raw_data.select_dtypes(include=['object', 'category']).columns

# Numerical Pipeline: Standardize data
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())  # Standardize the data
])

# Categorical Pipeline: One-hot encode data
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# Apply the preprocessor to the data
cleaned_data = preprocessor.fit_transform(raw_data)

# Encode the target data
le = LabelEncoder()
clean_targets = le.fit_transform(target_data)

# Convert the cleaned data to a DataFrame if needed
cleaned_data_df = pd.DataFrame(cleaned_data)
clean_targets_df = pd.DataFrame(clean_targets)

# Assign column names to the cleaned data
cleaned_data_df.columns = preprocessor.get_feature_names_out()
cleaned_data_df['target'] = clean_targets_df

# Display the cleaned data
print(cleaned_data_df.head())

# Save the cleaned data to a new CSV file
cleaned_data_df.to_csv(os.path.join(current_dir, data_dir, 'cirrhosis-CLEAN.csv'), index=False)