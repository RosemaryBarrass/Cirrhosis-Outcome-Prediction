import pandas as pd
import json
import matplotlib.pyplot as plt

def count_unique_entries(df, columns):    
    # Dictionary to store results
    unique_counts = {}
    
    # Loop through each specified column and count its unique values
    for column in columns:
        if column in df.columns:
            unique_counts[column] = df[column].value_counts().to_dict()
        else:
            unique_counts[column] = f"Column '{column}' not found in the data."
    
    return unique_counts

def count_missing_values(df):
    # Convert 'NA' strings to actual NaN values
    df.replace('NA', pd.NA, inplace=True)
    
    # Dictionary to store counts of missing values
    missing_counts = {}
    
    # Loop through each column in the DataFrame and count missing values
    for column in df.columns:
        missing_counts[column] = int(df[column].isna().sum())
    
    return missing_counts

if __name__ == '__main__':
    # Specify the CSV file path and columns to analyze
    csv_file_path = 'data/cirrhosis-RAW.csv'
    df = pd.read_csv(csv_file_path)
    columns_to_count = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

    # Get the counts of unique entries in the specified columns
    unique_entries_count = count_unique_entries(df, columns_to_count)
    missing_counts = count_missing_values(df)

    # Save the results
    with open('data_analysis/viz/kw/missing_counts.json', 'w') as json_file:
        json.dump(missing_counts, json_file, indent=4, sort_keys=True)

    with open('data_analysis/viz/kw/unique_entries_count.json', 'w') as json_file:
        json.dump(unique_entries_count, json_file, indent=4, sort_keys=True)

    df = pd.read_csv(csv_file_path)
    result = df.groupby('Status')['N_Days'].mean().to_dict()
    print("Average N_Days: ", result)

    # Calculate means and standard deviations
    grouped = df.groupby('Status')['N_Days']
    means = grouped.mean()
    errors = grouped.std()

    # Plotting
    fig, ax = plt.subplots()
    means.plot(kind='bar', yerr=errors, ax=ax, capsize=4)  # capsize specifies the width of the error caps
    ax.set_xlabel('Status')
    ax.set_ylabel('Average Number of Days')
    ax.set_title('Average Number of Days by Status')
    plt.savefig('data_analysis/viz/kw/days_vs_status.png')

    # Group by 'Drug' and 'Status' and count occurrences
    grouped = df.groupby(['Drug', 'Status']).size().unstack(fill_value=0)
    # Convert counts to percentages
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    # Plotting
    ax = percentages.plot(kind='bar', stacked=True, figsize=(10, 7))
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Drug')
    ax.set_title('Percentage of Final Status by Drug Treatment')

    # Adding percentage labels to each section of the stacked bars
    for rects in ax.patches:
        # Get x and height from the rectangle patch
        x = rects.get_x() + rects.get_width() / 2
        height = rects.get_height()
        y = rects.get_y() + height / 2

        # Only add text inside the bar if there's enough space (height is enough to be seen)
        if height > 0:
            ax.text(x, y, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    plt.xticks(rotation=45)  # Rotate drug names for better readability
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig('data_analysis/viz/kw/status_vs_drug.png')