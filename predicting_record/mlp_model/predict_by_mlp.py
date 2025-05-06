import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from submission_record import prepare_submission


def clean_input(arr):
    return pd.DataFrame(arr).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()


DATA_DIR = "../../data"
# train_dataset = "train_merged.csv"  # use the dataset which merge train.csv and train_2.csv
# train_dataset = "train.csv"  # use the original dataset train.csv
train_dataset = "train_raw.csv"  # use the dataset train.csv which is built by 'interpolate_and_split.py'

# Loading training dataset
train_df = pd.read_csv(os.path.join(DATA_DIR, train_dataset))

# Construct input and output datasets
# X_train = train_df.loc[:, [str(i) for i in range(-49, 1)]].to_numpy()
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy()

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
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()

# Check and clean missing values
X_test = clean_input(X_test)

# Data Normalization
X_test = scaler.transform(X_test)

# Split dataset for training and prediction
tX, vX, ty, vy = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
tX2, vX2, ty2, vy2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=42)

# Training MLP Model
print("Training MLP Model...")
model1 = MLPRegressor(
    hidden_layer_sizes=(512, 256),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
    verbose=True,
    alpha=1e-4
)
model1.fit(tX, ty)

model2 = MLPRegressor(
    hidden_layer_sizes=(512, 256),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
    verbose=True,
    alpha=1e-4
)
model2.fit(tX2, ty2)

# Check MSE
val_pred = model1.predict(vX)
mse = mean_squared_error(vy, val_pred)
print(f"MSE: {mse:.5f}")

# Prediction
y_pred = model1.predict(X_test)
X_test2 = scaler2.transform(y_pred)
y_pred_r_1_to_9 = model2.predict(X_test2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename1 = f"y_pred_mlp_{timestamp}.npy"
filename2 = f"y_pred_r_1_to_9_mlp_merged_{timestamp}.npy"
np.save(filename1, y_pred)
np.save(filename2, y_pred_r_1_to_9)

print(f"Prediction completed and the result has been saved to: y_pred_mlp_{timestamp}.npy, shape={y_pred.shape}")
print(f"Prediction completed and the result has been saved to: y_pred_r_1_to_9_mlp_merged_{timestamp}.npy, shape={y_pred_r_1_to_9.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "mlp")
