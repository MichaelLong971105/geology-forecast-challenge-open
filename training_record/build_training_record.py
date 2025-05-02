import os
import numpy as np
import pandas as pd
from glob import glob

# 设置路径
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))
RAW_DIR = os.path.join(DATA_DIR, "train_raw")
print("RAW_DIR =", RAW_DIR)
files = glob(os.path.join(RAW_DIR, "*.csv"))
print(f"✅ 找到 {len(files)} 个原始轨迹文件")

# 参数设置
INPUT_LENGTH = 300
OUTPUT_LENGTH = 300
WINDOW_SIZE = INPUT_LENGTH + OUTPUT_LENGTH
STRIDE = 20

X_all = []
y_all = []

print("▶ 开始处理 train_raw 中的轨迹文件...")

for file in files:
    try:
        df = pd.read_csv(file)
        # if df.shape[0] < WINDOW_SIZE:
        #     continue

        x = df['VS_APPROX_adjusted'].values
        z = df['HORIZON_Z_adjusted'].values

        # 插值：以 1ft 间隔构建 Z 曲线
        x_new = np.arange(x.min(), x.max(), step=1.0)
        if len(x_new) < WINDOW_SIZE:
            continue

        z_interp = np.interp(x_new, x, z)
        print(f"🔍 {os.path.basename(file)} 插值点数 = {len(x_new)}")

        count = 0
        # 滑窗提取样本
        for start in range(0, len(z_interp) - WINDOW_SIZE + 1, STRIDE):
            chunk = z_interp[start:start + WINDOW_SIZE]
            X = chunk[:INPUT_LENGTH]
            y = chunk[INPUT_LENGTH:]

            # 简单质量控制
            if np.isnan(X).any() or np.isnan(y).any():
                continue
            if np.std(X) < 0.01:  # 放宽过滤阈值
                continue

            X_all.append(X)
            y_all.append(y)
            count += 1

        print(f"✅ {os.path.basename(file)} 提取样本数: {count}")
    except Exception as e:
        print(f"⚠️ 处理 {file} 出错: {e}")

# 保存
X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)

np.save(os.path.join(BASE_DIR, "X_train.npy"), X_all)
np.save(os.path.join(BASE_DIR, "y_train.npy"), y_all)

print(f"✅ 样本提取完成: X={X_all.shape}, y={y_all.shape}")