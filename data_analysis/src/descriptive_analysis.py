import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current working directory
current_dir = os.getcwd()

# Specify some folder names
data_dir = 'data' 
viz_dir = os.path.join('data_analysis','viz')

# Specify the file name
file_name = 'cirrhosis-CLEAN.csv'

# Construct the full paths
file_path = os.path.join(current_dir, data_dir, file_name)
output_dir = os.path.join(current_dir, viz_dir)

# Load the dataset
data = pd.read_csv(file_path)

# Descriptive Statistics
print("Summary Statistics:")
print(data.describe())

# Missing Values
print("\nMissing values per column:")
print(data.isnull().sum())

# Visualizing Data
print("\nPlotting histograms:")
data.hist(figsize=(12, 10))
plt.suptitle('Histograms of Numerical Columns')
plt.savefig(os.path.join(output_dir, 'histogram_num.png'))
plt.close()

# Visualizing Categorical Data (Processed as Integer Values)
print("\nPlotting bar plots of categorical columns:")
categorical_columns = [col for col in data.columns if col.startswith('cat')]
for col in categorical_columns:
    plt.figure()
    sns.countplot(x=col, data=data)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    # Save the count plot as a PNG file
    plt.savefig(os.path.join(output_dir, f'countplot_{col}.png'))
    plt.close()

# Correlation Matrix
print("\nCorrelation matrix:")
corr_matrix = data.corr()
print(corr_matrix)

 # Visualizing Correlation Matrix
print("\nVisualizing correlation matrix:")
plt.figure(figsize=(50, 50))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# Save the heatmap as a PNG file
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()