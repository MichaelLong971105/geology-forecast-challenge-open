import os
import numpy as np
import pandas as pd
from submission_record import prepare_submission
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from datetime import datetime


def clean_input(X):
    return pd.DataFrame(X).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()


def find_best_k_metric(max_k, metric_list, X_train, y_train, n_splits=5):
    result_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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

    # 找出 MSE 最小的组合
    best_result = min(result_list, key=lambda x: x[2])
    print(f"\nBest k and metric: metric={best_result[0]}, k={best_result[1]}, MSE={best_result[2]:.5f}")

    return best_result


# 设置路径
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = "../../data"
train_dataset = "train_merged.csv"

# 加载合并训练集
train_df = pd.read_csv(os.path.join(DATA_DIR, train_dataset))

# 构造输入输出
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy(dtype=np.float32)

# 检查并清洗缺失值
X_train = clean_input(X_train)
y_train = clean_input(y_train)

# 加载测试集
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)

# 缺失值处理（插值 + 前后向填充）
X_test = clean_input(X_test)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

final_k_metric = find_best_k_metric(100, ['euclidean', 'manhattan', 'cosine'], X_train, y_train)

# 训练 KNN 模型（使用合适的 k）
print("正在训练 KNN 模型...")
model = KNeighborsRegressor(n_neighbors=final_k_metric[1], weights='distance', metric=final_k_metric[0])
# model = KNeighborsRegressor(n_neighbors=33, weights='distance', metric='manhattan')
model.fit(X_train, y_train)

# 预测
print("正在预测测试集...")
y_pred = model.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_knn_merged_{timestamp}.npy"
np.save(filename, y_pred)

print(f"KNN模型预测完成，预测结果已保存到 y_pred_knn_merged_{timestamp}.npy，shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "knn")
