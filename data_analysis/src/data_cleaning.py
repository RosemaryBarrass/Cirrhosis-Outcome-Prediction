import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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

### Data Cleaning ###

# Change age data to more interpretable year values instead of days
raw_data['Age'] = raw_data['Age']/365

# Check for missing values
print(raw_data.isnull().sum())

# Define numerical and categorical columns
# 'ID' an 'N_Days' columns are not included
numerical_columns = raw_data[['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']].columns
onehot_columns = raw_data[['Drug','Sex', 'Ascites', 'Hepatomegaly', 'Spiders']].columns
ordinal_columns = raw_data[['Status', 'Edema', 'Stage']].columns

# Categorical Pipeline: One-hot encode data
onehot_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),  # One-hot encode 'Y' 'N' data
])

# Categorical Pipeline: ordinal encode data
ordinal_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(handle_unknown='error')),  # ordinal encode integer categorical data
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer([
    ('numerical', 'passthrough', numerical_columns),    # pass through numerical data, untouched
    ('onehot', onehot_pipeline, onehot_columns),    # one-hot encode alll binary data
    ('ordinal', ordinal_pipeline, ordinal_columns)  # ordinal encode data with stages
])

# Apply the preprocessor to the data
cleaned_data = preprocessor.fit_transform(raw_data)

# Convert the cleaned data to a DataFrame if needed
cleaned_data_df = pd.DataFrame(cleaned_data)

# Assign column names to the cleaned data
cleaned_data_df.columns = preprocessor.get_feature_names_out()

# Display the cleaned data
print(cleaned_data_df.head())

# Save the cleaned data to a new CSV file
cleaned_data_df.to_csv(os.path.join(current_dir, data_dir, 'cirrhosis-CLEAN.csv'), index=False)