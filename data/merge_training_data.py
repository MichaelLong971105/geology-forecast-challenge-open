import pandas as pd

# 加载两个训练集
train1 = pd.read_csv("train.csv")
train2 = pd.read_csv("train_2.csv")

# 合并
merged_train = pd.concat([train1, train2], ignore_index=True)

# 去重（可选）
merged_train.drop_duplicates(subset=merged_train.columns.difference(['geology_id']), inplace=True)

# 保存
merged_train.to_csv("train_merged.csv", index=False)
print(f"✅ 合并完成，保存为 train_merged.csv，样本数 = {merged_train.shape[0]}")
