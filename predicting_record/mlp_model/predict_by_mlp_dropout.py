import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from submission_record import prepare_submission

# --- 参数设置 ---
SEED = 42
BATCH_SIZE = 128
EPOCHS = 300
DROPOUT_RATE = 0.2
REALIZATION_COUNT = 10
LEARNING_RATE = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 数据加载与预处理 ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data"))
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_merged.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X = train_df.loc[:, [str(i) for i in range(-299, 1)]].interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()
y = train_df.loc[:, [str(i) for i in range(1, 301)]].interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()

X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# --- 构建神经网络模型 ---
class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim=300, output_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# --- 训练函数 ---
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()


# --- 构造训练集 ---
tX, vX, ty, vy = train_test_split(X, y, test_size=0.2, random_state=SEED)
train_dataset = TensorDataset(torch.tensor(tX, dtype=torch.float32), torch.tensor(ty, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 训练模型 ---
model = MCDropoutMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
train_model(model, train_loader, optimizer, criterion)


# --- MCDropout 推理函数 ---
def predict_mc(model, X_np, n_samples=REALIZATION_COUNT):
    model.eval()
    model.train()  # 开启 dropout
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_sample = model(X_tensor).numpy()
            preds.append(y_sample)
    return np.stack(preds)  # shape = [n_samples, 524, 300]


# --- 执行 MCDropout 推理 ---
y_samples = predict_mc(model, X_test, n_samples=REALIZATION_COUNT)
y_pred = y_samples[0]  # realization_0

# --- 保存主预测结果 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_mlp_mc_{timestamp}.npy"
np.save(filename, y_pred)
print(f"✅ MLP MCDropout 主预测保存至 y_pred_mlp_mc_{timestamp}.npy, shape={y_pred.shape}")

# # --- 保存全部10组用于生成 submission ---
# for r in range(REALIZATION_COUNT):
#     path = os.path.join("", f"y_pred_mlp_dropout_r{r}_{timestamp}.npy")
#     np.save(path, y_samples[r])

prepare_submission.prepareSubmission(y_samples, "../../submission_record", "mcdropout")




