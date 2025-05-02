import os
import pandas as pd
import numpy as np

# 设置路径
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))
train_path = os.path.join(DATA_DIR, "train_merged.csv")

# 加载数据
df = pd.read_csv(train_path)

# 提取输入（-299 ~ 0）
X = df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)

# 提取 10 个目标输出（realization_0 到 realization_9）
y_list = []
y_list.append(df.loc[:, [str(i) for i in range(1, 301)]].to_numpy(dtype=np.float32))

for r in range(1, 10):
    cols = [f"r_{r}_pos_{i}" for i in range(1, 301)]
    y_r = df.loc[:, cols].to_numpy(dtype=np.float32)
    y_list.append(y_r)

# 扩展训练集
X_aug = np.repeat(X, 10, axis=0)
y_aug = np.vstack(y_list)

# 保存
np.save("X_train_augmented.npy", X_aug)
np.save("y_train_augmented.npy", y_aug)

print(f"✅ 增强训练集生成成功：X={X_aug.shape}, y={y_aug.shape}")
