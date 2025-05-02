import os
import time
import numpy as np
import pandas as pd
from submission_record import prepare_submission
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

def clean_input(X):
    return pd.DataFrame(X).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()

def find_best_k_metric(k, metric_list, X_train_sub, y_train_sub, y_val):
    result_list = []
    for k in range(1, k+1):
        for metric in metric_list:
            model = KNeighborsRegressor(n_neighbors=k, weights='distance', metric=metric)
            model.fit(X_train_sub, y_train_sub)
            y_pred_val = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred_val)
            result_list.append((metric, k, mse))

    min_mse = result_list[0][2]
    i = 0
    for result in result_list:
        if result[2] < min_mse:
            i = result_list.index(result)

    print(result_list[i])
    return result_list[i]

# 设置路径
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = "../../data"

# 加载合并训练集
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_2.csv"))

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

# final_k_metric = find_best_k_metric(100, ['euclidean', 'manhattan', 'cosine'], X_train_sub, y_train_sub, y_val)

# 训练 KNN 模型（使用合适的 k）
print("▶ 正在训练 KNN 模型...")
# model = KNeighborsRegressor(n_neighbors=final_k_metric[1], weights='distance', metric=final_k_metric[0])
model = KNeighborsRegressor(n_neighbors=8, weights='distance', metric='euclidean')
model.fit(X_train, y_train)

# 预测
print("▶ 正在预测测试集...")
y_pred = model.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_knn_merged_{timestamp}.npy"
np.save(filename, y_pred)

print(f"✅ KNN 模型预测完成，预测结果已保存到 y_pred_knn_merged_{timestamp}.npy，shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "knn")




#
# import os
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
#
# # 加载训练数据（来自 train_raw 生成的增强版）
# X_train = np.load("../../training_record/X_train.npy")
# y_train = np.load("../../training_record/y_train.npy")
#
# # 加载测试数据
# test_df = pd.read_csv("../../data/test.csv")
# X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy(dtype=np.float32)
#
# # 缺失值处理
# X_test = pd.DataFrame(X_test).interpolate(axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1).to_numpy()
#
# # 训练并预测
# print("▶ 正在训练 KNN 模型...")
# model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean')
# model.fit(X_train, y_train)
#
# print("▶ 正在预测测试集...")
# y_pred = model.predict(X_test)
# np.save("y_pred_knn.npy", y_pred)
#
# print(f"✅ KNN from raw_v2 模型预测完成，结果保存至 y_pred_knn.npy，shape={y_pred.shape}")



