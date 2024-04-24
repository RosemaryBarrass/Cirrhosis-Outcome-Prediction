import numpy as np
import pandas as pd
from unique_counts import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

N_ESTIMATORS_GRID = [25, 50, 100, 200, 400, 800, 1600, 3200]
LEARNING_RATE_GRID = [0.01, 0.05, 0.1, 0.2]
KNN_IMPUTE_GRID = [5]


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


def get_accuracy(data_id, model_type, n_estimators_grid, learning_rate_grid):
    data = pd.read_csv('data/cirrhosis-knnimputed-k{}.csv'.format(data_id))
    X = np.array(data.drop(columns=['ordinal__Status']))
    y = np.array(data['ordinal__Status'])

    acc = np.empty((len(n_estimators_grid), len(learning_rate_grid)))
    for i, n_estimators in enumerate(n_estimators_grid):
        for j, learning_rate in enumerate(learning_rate_grid):
            if model_type == 'gradient_boost':
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=0)
            elif model_type == 'adaboost':
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
            elif model_type == 'catboost':
                model = CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, verbose=False, random_state=0)
            elif model_type == 'lgbm':
                model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0, verbose=0)
            else:
                raise ValueError("Unsupported model type: {}".format(model_type))
            acc[i, j] = run_model(model, X, y)
            print("{} n_estimators {} learning rate done".format(n_estimators, learning_rate))
        
    return acc

def get_output_files(model_type):
    acc_results = []
    for i, k in enumerate(KNN_IMPUTE_GRID):
        acc = get_accuracy(k, model_type, N_ESTIMATORS_GRID, LEARNING_RATE_GRID)
        acc_results.append(acc)

        # Convert accuracy matrix to DataFrame and save to CSV
        df_acc = pd.DataFrame(acc, columns=[f"LR={lr}" for lr in LEARNING_RATE_GRID], index=[f"NE={ne}" for ne in N_ESTIMATORS_GRID])
        df_acc.to_csv(f'output/{model_type}/acc_matrix_for_k{k}.csv', index=True)

    return

# get_output_files('gradient_boost')
# get_output_files('adaboost')
# get_output_files('catboost')
# get_output_files('xgboost')
get_output_files('lgbm')

