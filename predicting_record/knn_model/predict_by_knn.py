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
train_dataset = "train.csv"  # use the original dataset train.csv
# train_dataset = "train_raw.csv"  # use the dataset train.csv which is built by 'interpolate_and_split.py'

# Loading training dataset
train_df = pd.read_csv(os.path.join(DATA_DIR, train_dataset))

# Construct input and output datasets
# X_train = train_df.loc[:, [str(i) for i in range(-49, 1)]].to_numpy(dtype=np.float32)
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy(dtype=np.float32)

X_train2 = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy(dtype=np.float32)
y_train2 = train_df.loc[:, [f"r_{r}_pos_{i}" for r in range (1, 10) for i in range(1, 301)]].to_numpy(dtype=np.float32)

# Check and clean missing values
X_train = clean_input(X_train)
y_train = clean_input(y_train)

X_train2 = clean_input(X_train2)
y_train2 = clean_input(y_train2)

# Data Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)

# Loading testing dataset
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)

# Check and clean missing values
X_test = clean_input(X_test)

# Data Normalization
X_test = scaler.transform(X_test)

# Using find_best_parameter function to find the combination with the smallest MSE
best_parameter1 = find_best_parameter(100, ['euclidean', 'manhattan', 'cosine'], X_train, y_train)
best_parameter2 = find_best_parameter(100, ['euclidean', 'manhattan', 'cosine'], X_train2, y_train2)

# Training KNN Model with the combination with the smallest MSE
print("Training KNN Model...")
model1 = KNeighborsRegressor(n_neighbors=best_parameter1[1], weights='distance', metric=best_parameter1[0])
model1.fit(X_train, y_train)

model2 = KNeighborsRegressor(n_neighbors=best_parameter2[1], weights='distance', metric=best_parameter2[0])
model2.fit(X_train2, y_train2)

# Prediction
print("Predicting...")
y_pred = model1.predict(X_test)
X_test2 = scaler2.transform(y_pred)
y_pred_r_1_to_9 = model2.predict(X_test2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename1 = f"y_pred_knn_merged_{timestamp}.npy"
filename2 = f"y_pred_r_1_to_9_knn_merged_{timestamp}.npy"
np.save(filename1, y_pred)
np.save(filename2, y_pred_r_1_to_9)

print(f"Prediction completed and the result has been saved to: y_pred_knn_merged_{timestamp}.npy, shape={y_pred.shape}")
print(f"Prediction completed and the result has been saved to: y_pred_r_1_to_9_knn_merged_{timestamp}.npy, shape={y_pred_r_1_to_9.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "knn", y_pred_r_1_to_9)
