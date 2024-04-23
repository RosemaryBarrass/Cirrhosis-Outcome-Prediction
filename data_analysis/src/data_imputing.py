import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

# Get the current working directory
current_dir = os.getcwd()

# Specify some folder names
data_dir = 'data' 

# Specify the file name
feat_file = 'cirrhosis-CLEAN.csv'

# Construct the full paths
file_path = os.path.join(current_dir, data_dir, feat_file)

# Load the dataset
data = pd.read_csv(file_path)

# Remove the data of the patients not included in the clinical trial
data = data[data['bin__Ascites'].isin([0, 1])]

k_values = list(range(2,10))

# Loop to impute with different K values
for k in k_values:
    # Handle any other NaN values by imputing them
    imputer = KNNImputer()
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed.to_csv(os.path.join(data_dir, f'cirrhosis-knnimputed-k{k}.csv'), index=False)

# Compare to simple mean imputer
simple_imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(simple_imputer.fit_transform(data), columns=data.columns)
data_imputed.to_csv(os.path.join(data_dir, f'cirrhosis-meanimputed.csv'), index=False)

# Compare to EXPERIMENTAL iterative imputer
iterative_imputer = IterativeImputer()
data_imputed = pd.DataFrame(iterative_imputer.fit_transform(data), columns=data.columns)
data_imputed.to_csv(os.path.join(data_dir, f'cirrhosis-iterativeimputed.csv'), index=False)