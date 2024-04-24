import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

methods = ['adaboost', 'catboost', 'gradient_boost', 'lgbm', 'xgboost']
dataframes = []

# Read dataframes and store them in a list
for method in methods:
    file_path = f'output/{method}/acc_matrix_for_k5.csv'
    df = pd.read_csv(file_path, index_col=0)
    dataframes.append(df)

# Find the overall minimum and maximum values for consistent color scale across heatmaps
min_val = min(df.values.min() for df in dataframes)
max_val = max(df.values.max() for df in dataframes)

plt.figure(figsize=(12, 40))  # Adjust size based on the number of methods

# Plotting each method's heatmap with best accuracy in the title
for i, method in enumerate(methods):
    best_accuracy = dataframes[i].max().max()  # Max value in each DataFrame
    plt.subplot(len(methods), 1, i+1)  # Creating a subplot for each CSV
    heatmap = sns.heatmap(dataframes[i], annot=True, fmt=".3f", cmap='viridis', vmin=min_val, vmax=max_val)
    plt.title(f'Accuracy for {method} (Best: {best_accuracy:.3f})')  # Include best accuracy in title
    plt.xlabel('Learning Rate (LR)')
    plt.ylabel('Number of Estimators (NE)')

plt.tight_layout()  # Adjusts subplots to fit into figure area.
plt.savefig('output/boosting_heatmaps.png')  # Save the full figure
plt.show()  # Show the figure in a viewer
