import numpy as np
import pandas as pd
from unique_counts import *
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

C_VALUES = [0.01, 0.1, 1, 10, 100]

def run_model(model, X, y):
    kf = KFold(n_splits=5)
    acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)  # Training the model
        predictions = model.predict(X_test)  # Predicting the test set
        acc.append(accuracy_score(y_test, predictions))  # Evaluating accuracy

    return np.mean(acc)

data = pd.read_csv("data/cirrhosis-knnimputed-k5.csv")
X = data.drop(columns=['ordinal__Status'])
y = data['ordinal__Status']

# Step 3: Dynamically identify columns
numerical_cols = [col for col in X.columns if col.startswith('numerical__')]
categorical_cols = [col for col in X.columns if col.startswith('bin__') or col.startswith('ordinal__')]

# Step 4: Create a Column Transformer to apply scaling to numerical and one-hot encoding to categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Step 5: Apply transformations
X_transformed = preprocessor.fit_transform(X)

# Dictionary to store accuracies for different kernels
kernel_accuracies = {'linear': [], 'rbf': [], 'poly': []}

# Test each kernel and C value
for kernel in kernel_accuracies.keys():
    for C in C_VALUES:
        model = SVC(kernel=kernel, C=C, decision_function_shape='ovo', random_state=0)
        accuracy = run_model(model, X_transformed, y)
        kernel_accuracies[kernel].append(accuracy)

# Plotting the results
plt.figure(figsize=(10, 8))
for kernel, accuracies in kernel_accuracies.items():
    plt.plot(C_VALUES, accuracies, label=f'{kernel} kernel', marker='o')

plt.title('SVM Accuracy vs Regularization Term C for Different Kernels')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig("output/svm/svm_performances.png")

# Sort the DataFrame by the absolute values of 'Importance' in descending order
data = pd.read_csv("data_analysis/viz/kw/truncated_rows_feature_importance.csv")

data['abs_importance'] = data['Importance'].abs()
data_sorted = data.sort_values(by='abs_importance', ascending=False).drop('abs_importance', axis=1)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(data_sorted['Feature'], data_sorted['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances Sorted by Absolute Value')
plt.gca().invert_yaxis()  # Invert y-axis to have the largest at top
plt.savefig("output/svm/svm_feature_importances.png")

