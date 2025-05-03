import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from submission_record import prepare_submission


# 清洗函数
def clean_input(arr):
    return pd.DataFrame(arr).interpolate(axis=1).bfill(axis=1).ffill(axis=1).to_numpy()


def find_best_parameter(hidden_layer_sizes, activations, alphas, X_train, y_train, n_splits=5):
    result_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("正在寻找MSE最小的参数组合")
    for hiddenLayer in hidden_layer_sizes:
        for activation in activations:
            for alpha in alphas:
                mse_fold = []
                for train_index, val_index in kf.split(X_train):
                    X_tr, X_val = X_train[train_index], X_train[val_index]
                    y_tr, y_val = y_train[train_index], y_train[val_index]
                    model = MLPRegressor(
                        hidden_layer_sizes=hiddenLayer,
                        activation=activation,
                        alpha=alpha,
                        solver='adam',
                        learning_rate_init=0.001,
                        max_iter=2000,
                        random_state=42
                    )
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    mse_fold.append(mse)
                avg_mse = np.mean(mse_fold)
                result_list.append((hiddenLayer, activation, alpha, avg_mse))

    best_result = min(result_list, key=lambda x: x[2])
    print(f"\nBest Parameter: hidden_layer_sizes={best_result[0]}, activations={best_result[1]}, alphas={best_result[2]}, MSE={best_result[3]:.5f}")

    return best_result

# 路径设置
DATA_DIR = "../../data"
train_dataset = "train_merged.csv"

# 加载训练数据
train_df = pd.read_csv(os.path.join(DATA_DIR, train_dataset))

# 构造输入输出
X_train = train_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
y_train = train_df.loc[:, [str(i) for i in range(1, 301)]].to_numpy()

# 检查并清洗缺失值
X_train = clean_input(X_train)
y_train = clean_input(y_train)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 加载测试集
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
X_test = test_df.loc[:, [str(i) for i in range(-299, 1)]].to_numpy()
X_test = clean_input(X_test)


X_test = scaler.transform(X_test)

# 执行评估函数
# best_parameter = find_best_parameter([(512, 256), (512, 256, 128), (256, 128)], ['tanh', 'relu'], [1e-4, 1e-3], X_train, y_train)

# 使用全量训练数据重新训练并预测测试
# print("▶ 使用全量训练集重新训练 MLP 模型...")
# model = MLPRegressor(
#     hidden_layer_sizes=best_parameter[0],
#     activation=best_parameter[1],
#     alpha=best_parameter[2],
#     solver='adam',
#     learning_rate_init=0.001,
#     max_iter=2000,
#     random_state=42
# )

model2 = MLPRegressor(
    hidden_layer_sizes=(512, 256),
    activation='tanh',
    alpha=1e-4,
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)
model2.fit(X_train, y_train)

# 预测测试集
# y_pred = model.predict(X_test)
y_pred = model2.predict(X_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_mlp_{timestamp}.npy"
np.save(filename, y_pred)
print(f"✅ MLP 模型预测完成，预测结果已保存到 y_pred_mlp_{timestamp}.npy, shape={y_pred.shape}")
prepare_submission.prepareSubmission(y_pred, "../../submission_record", "mlp")

