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
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy()

# Check and clean missing values
X_train = clean_input(X_train)
y_train = clean_input(y_train)

# Data Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Loading testing dataset
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()

# Check and clean missing values
X_test = clean_input(X_test)

# Data Normalization
X_test = scaler.transform(X_test)

# Split dataset for training and prediction
tX, vX, ty, vy = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Training MLP Model
print("Training MLP Model...")
model = MLPRegressor(
    hidden_layer_sizes=(512, 256),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
    verbose=True,
    alpha=1e-4
)
model.fit(tX, ty)

# Check MSE
val_pred = model.predict(vX)
mse = mean_squared_error(vy, val_pred)
print(f"MSE: {mse:.5f}")

# Prediction
y_pred = model.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_mlp_{timestamp}.npy"
np.save(filename, y_pred)

print(f"Prediction completed and the result has been saved to: y_pred_mlp_{timestamp}.npy, shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "mlp")
