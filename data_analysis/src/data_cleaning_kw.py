import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Function to replace binary columns according to mappings
def replace_binary_columns(df, mappings):
    for column in df.columns:
        df[column] = df[column].map(mappings)
    return df

# Function to apply one-hot encoding to categorical columns
def encode_categorical_columns(df):
    encoder = OneHotEncoder(sparse=False)
    encoded_df = pd.DataFrame(encoder.fit_transform(df))
    encoded_df.columns = encoder.get_feature_names_out(df.columns)
    return encoded_df


file_name = 'data/cirrhosis-RAW.csv'
raw_data = pd.read_csv(file_name)
# Convert 'NA' strings to actual NaN values
raw_data.replace('NA', pd.NA, inplace=True)

target_data = raw_data['Status']
misc = raw_data[['ID', 'N_Days']]
binaries = raw_data[['Ascites', 'Hepatomegaly', 'Sex', 'Spiders', 'Drug']]
categoricals = raw_data[['Edema', 'Stage']]
numericals = raw_data[['Age', 'Bilirubin', 'Cholesterol', 
                       'Albumin', 'Copper', 'Alk_Phos', 
                       'SGOT', 'Tryglicerides', 
                       'Platelets' ,'Prothrombin']]

binary_mappings = {
    "M": 1,
    "F": 0,
    "Y": 1,
    "N": 0,
    "D-penicillamine": 1,
    "Placebo": 0
}

binaries = replace_binary_columns(binaries, binary_mappings)
categoricals_encoded = encode_categorical_columns(categoricals)

final_data = pd.concat([misc, numericals, binaries, categoricals_encoded, target_data], axis=1)
final_data.to_csv('data/cirrhosis-CLEAN-kw.csv', index=False)

# misc: cols 0-1
# numericals: cols 2-11
# binaries: cols 12-24 (includes Stage_nan)
# target: col 25
