import os
import pandas as pd
import numpy as np
from datetime import datetime

def prepareSubmission(y_pred, save_dir, model_type="knn", y_pred_r_1_to_9=None):
    # 设置路径
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

    # 加载样本提交格式
    sample_df = pd.read_csv(sample_path)
    geology_ids = sample_df['geology_id'].values

    # 创建提交 DataFrame
    submission = pd.DataFrame({'geology_id': geology_ids})

    # 添加 realization 0（主模型输出）
    for i in range(1, 301):
        submission[str(i)] = y_pred[:, i - 1]

    # 添加 realization 1~9
    if y_pred_r_1_to_9 is not None:
        print("✅ 使用 KNN 模型 2 输出的 r_1~r_9")
        for r in range(1, 10):
            cols = [f"r_{r}_pos_{i}" for i in range(1, 301)]
            start = (r - 1) * 300
            end = r * 300
            submission[cols] = y_pred_r_1_to_9[:, start:end]
    else:
        print("⚠️ 未提供模型2结果，使用高斯噪声生成 r_1~r_9")
        for r in range(1, 10):
            noise = np.random.normal(loc=0.0, scale=0.05, size=y_pred.shape)
            noisy_pred = y_pred + noise
            for i in range(1, 301):
                submission[f"r_{r}_pos_{i}"] = noisy_pred[:, i - 1]

    # 统一列顺序为 sample_submission 的列顺序
    sample_df.columns = sample_df.columns.astype(str)
    submission.columns = submission.columns.astype(str)
    submission = submission[sample_df.columns]

    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"submission_{model_type}_{timestamp}.csv")
    submission.to_csv(filename, index=False)
    print(f"✅ 提交文件保存至: {filename}")
