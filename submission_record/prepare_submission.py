import os
import pandas as pd
import numpy as np
from datetime import datetime


def prepareSubmission(y_pred, path, model_type):
    # 设置路径
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))

    # 加载 geoid 顺序
    sample_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    geology_ids = sample_df['geology_id'].values

    # 构造提交文件
    submission = pd.DataFrame({'geology_id': geology_ids})

    # 添加 realization 0（一次性构建）
    if model_type == "mcdropout":
        for i in range(1, 301):
            submission[str(i)] = y_pred[0, :, i - 1]  # realization 0
        for r in range(1, 10):
            col_names = [f"r_{r}_pos_{i}" for i in range(1, 301)]
            df_r = pd.DataFrame(y_pred[r], columns=col_names)
            submission = pd.concat([submission, df_r], axis=1)
    else:
        real0 = pd.DataFrame(y_pred, columns=[str(i) for i in range(1, 301)])
        submission = pd.concat([submission, real0], axis=1)
        # 添加 realization 1~9（一次性添加每组列）
        for r in range(1, 10):
            noise = np.random.normal(loc=0.0, scale=0.05, size=y_pred.shape)
            noisy_pred = y_pred + noise
            col_names = [f"r_{r}_pos_{i}" for i in range(1, 301)]
            df_realization = pd.DataFrame(noisy_pred, columns=col_names)
            submission = pd.concat([submission, df_realization], axis=1)

    # 保证列顺序与 sample_submission 一致
    submission = submission[sample_df.columns]

    # 添加时间戳并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{path}/submission_{model_type}_{timestamp}.csv"
    submission.to_csv(filename, index=False)

    print(f"✅ 提交文件已保存为 {filename}")
