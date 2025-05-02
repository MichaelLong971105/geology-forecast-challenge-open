import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from submission_record import prepare_submission

# 缺失值处理函数
def clean_input(arr):
    return pd.DataFrame(arr).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()

# 路径设置
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = "../../data"

# 加载训练数据
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_merged.csv"))
X = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
y = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy()

X = clean_input(X)
y = clean_input(y)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分验证集
tX, vX, ty, vy = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 MLP 模型
print("▶ 正在训练 MLP 模型...")
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

# 验证 MSE
val_pred = model.predict(vX)
mse = mean_squared_error(vy, val_pred)
print(f"🔍 验证集 MSE: {mse:.5f}")

# 加载测试数据
X_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = X_test.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
X_test = clean_input(X_test)
X_test = scaler.transform(X_test)

# 预测测试集
y_pred = model.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_mlp_{timestamp}.npy"
np.save(filename, y_pred)
print(f"✅ MLP 模型预测完成，预测结果已保存到 y_pred_mlp_{timestamp}.npy, shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "mlp")
