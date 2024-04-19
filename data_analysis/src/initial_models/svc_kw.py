# misc: cols 0-1
# numericals: cols 2-11
# binaries: cols 12-24 
# target: col 25

import numpy as np
import pandas as pd
from unique_counts import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def train_evaluate_save_model(X, y, csv_title, numerical_cols, binary_cols):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a ColumnTransformer for scaling numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'  # Leave binary features unchanged
    )

    # Create an SVC model
    svc = SVC(kernel='linear', random_state=42)

    # Create a pipeline that first transforms the data and then fits the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', svc)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = pipeline.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall:.2f}')

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Extract and save feature importances if the model is linear SVC
    if 'linear' in svc.kernel:
        coefficients = svc.coef_.ravel()  # Flatten the coefficients if necessary
        feature_names = list(numerical_cols) + list(binary_cols)
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
        feature_importances['Absolute Importance'] = feature_importances['Importance'].abs()
        feature_importances = feature_importances.sort_values(by='Absolute Importance', ascending=False).drop('Absolute Importance', axis=1)
        feature_importances.to_csv(csv_title, index=False)
    
    return svc

df = pd.read_csv('data/cirrhosis-CLEAN-kw.csv')
df = df[df['Status'] != 'CL']

"""
df = df.dropna()

columns_to_count = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
                    'Edema_N', 'Edema_S', 'Edema_Y', 'Stage_1.0', 
                    'Stage_2.0', 'Stage_3.0', 'Stage_4.0']

print(count_unique_entries(df, columns_to_count))

label_mapping = {'C': 0, 'D': 1}
df['Status'] = df['Status'].map(label_mapping)
numerical_cols = df.columns[2:12] 
binary_cols = df.columns[12:25]   
target_col = df.columns[25] 

# Split data into features and target
X = df.drop(columns=['ID', 'N_Days', 'Status'])
y = df[target_col]

# train_evaluate_save_model(X, y, 'data_analysis/viz/kw/truncated_rows_feature_importance.csv', numerical_cols, binary_cols)
"""

label_mapping = {'C': 0, 'D': 1}
df['Status'] = df['Status'].map(label_mapping)
cols_to_exclude = {'Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides'}
numerical_cols = list(set(df.columns[2:12]) - cols_to_exclude)
binary_cols = list(set(df.columns[12:25]) - cols_to_exclude)  
target_col = df.columns[25] 

# Split data into features and target
columns_to_drop = ['ID', 'N_Days', 'Status'] + list(cols_to_exclude)
X = df.drop(columns=columns_to_drop)
y = df[target_col]
df = pd.concat([X, y], axis=1)
df = df.dropna()
X = df.drop(columns=['Status'])
y = df['Status']
train_evaluate_save_model(X, y, 'data_analysis/viz/kw/truncated_cols_feature_importance.csv', numerical_cols, binary_cols)
