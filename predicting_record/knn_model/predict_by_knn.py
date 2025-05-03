import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from datetime import datetime

from submission_record import prepare_submission


def clean_input(X):
    return pd.DataFrame(X).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()


def find_best_parameter(max_k, metric_list, X_train, y_train, n_splits=5):
    result_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("Looking for the parameter combination with the smallest MSE...")
    for k in range(1, max_k + 1):
        for metric in metric_list:
            mse_fold = []
            for train_index, val_index in kf.split(X_train):
                X_tr, X_val = X_train[train_index], X_train[val_index]
                y_tr, y_val = y_train[train_index], y_train[val_index]
                model = KNeighborsRegressor(n_neighbors=k, weights='distance', metric=metric)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mse_fold.append(mse)

            avg_mse = np.mean(mse_fold)
            result_list.append((metric, k, avg_mse))

    # Find the combination with the smallest MSE
    best_result = min(result_list, key=lambda x: x[2])
    print(f"\nBest Parameter: metric={best_result[0]}, k={best_result[1]}, MSE={best_result[2]:.5f}")

    return best_result


DATA_DIR = "../../data"
# train_dataset = "train_merged.csv"  # use the dataset which merge train.csv and train_2.csv
# train_dataset = "train.csv"  # use the original dataset train.csv
train_dataset = "train_raw.csv"  # use the dataset train.csv which is built by 'interpolate_and_split.py'

# Loading training dataset
train_df = pd.read_csv(os.path.join(DATA_DIR, train_dataset))

# Construct input and output datasets
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy(dtype=np.float32)

# Check and clean missing values
X_train = clean_input(X_train)
y_train = clean_input(y_train)

# Data Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Loading testing dataset
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)

# Check and clean missing values
X_test = clean_input(X_test)

# Data Normalization
X_test = scaler.transform(X_test)

# Using find_best_parameter function to find the combination with the smallest MSE
best_parameter = find_best_parameter(100, ['euclidean', 'manhattan', 'cosine'], X_train, y_train)

# Training KNN Model with the combination with the smallest MSE
print("Training KNN Model...")
model = KNeighborsRegressor(n_neighbors=best_parameter[1], weights='distance', metric=best_parameter[0])
model.fit(X_train, y_train)

# Prediction
print("Predicting...")
y_pred = model.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_knn_merged_{timestamp}.npy"
np.save(filename, y_pred)

print(f"Prediction completed and the result has been saved to: y_pred_knn_merged_{timestamp}.npy, shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "knn")
