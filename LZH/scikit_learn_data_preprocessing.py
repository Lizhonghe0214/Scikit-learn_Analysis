"""
sklearn库实操：
数据预处理
"""

import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

print(sklearn.__version__)  # 查看版本

# 核心API：估计器、转换器、预测器
# 1.数据清洗
data = pd.DataFrame({
    'A': [1, 1, np.nan, 4],
    'B': [5, 5, 7, 8],
    'C': [9, 9, 11, 12]
})
# 删除重复行
data = data.drop_duplicates()
# 填充缺失值
data = data.fillna(data.mean())
print(data)

# 2.特征提取与转换
text_data = ["hello world", "hello everyone", "world of programming"]
vectorizer = CountVectorizer()
# 转换文本数据为词频矩阵
X = vectorizer.fit_transform(text_data)
print(X.toarray())

# 3.标准化与归一化
data = [[1, 2], [2, 3], [3, 4]]
# 标准化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
# 归一化
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)
print("Standardized data:\n", standardized_data)
print("Normalized data:\n", normalized_data)

# 4.缺失值处理
data = [[1, 2], [np.nan, 3], [7, 6]]
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data)
print(imputed_data)

# 5.数据集划分
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training data:\n", X_train, y_train)
print("Testing data:\n", X_test, y_test)
