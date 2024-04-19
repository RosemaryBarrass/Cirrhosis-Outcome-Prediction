import pandas as pd
import json

def count_unique_entries(csv_file, columns):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Dictionary to store results
    unique_counts = {}
    
    # Loop through each specified column and count its unique values
    for column in columns:
        if column in df.columns:
            unique_counts[column] = df[column].value_counts().to_dict()
        else:
            unique_counts[column] = f"Column '{column}' not found in the data."
    
    return unique_counts

def count_missing_values(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
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
    columns_to_count = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

    # Get the counts of unique entries in the specified columns
    unique_entries_count = count_unique_entries(csv_file_path, columns_to_count)
    missing_counts = count_missing_values(csv_file_path)

    # Save the results
    with open('data_analysis/viz/kw/missing_counts.json', 'w') as json_file:
        json.dump(missing_counts, json_file, indent=4, sort_keys=True)

    with open('data_analysis/viz/kw/unique_entries_count.json', 'w') as json_file:
        json.dump(unique_entries_count, json_file, indent=4, sort_keys=True)

    df = pd.read_csv(csv_file_path)
    result = df.groupby('Status')['N_Days'].mean().to_dict()
    print("Average N_Days: ", result)