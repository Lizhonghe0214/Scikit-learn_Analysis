"""
sklearn库实操：
监督学习
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# 1.线性模型
# 使用线性回归进行房价预测
X = np.array([[800], [1000], [1200], [1400], [1600], [1800], [2000], [2200], [2400]])
y = np.array([[150000], [175020], [198000], [224000], [250010], [280000], [300000], [330040], [349950]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差为：{mse}")
# 可视化
plt.scatter(X_test, y_test, color='blue', label='real housing price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='forecast housing price')
plt.xlabel('floor space(sq ft)')
plt.ylabel('housing price($)')
plt.legend()
plt.show()

# 2.支持向量机
# 使用线性SVM进行二分类
X, y = make_classification(n_samples=100, n_redundant=0, n_repeated=0, n_features=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 打印混淆矩阵
print(confusion_matrix(y_test, y_pred))
# 打印分类报告
print(classification_report(y_test, y_pred))

# 3.决策树与随机森林
# 使用决策树进行分类
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率为：{accuracy:.2f}")
# 使用随机森林进行分类
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率为：{accuracy:.2f}")

# 4.神经网络基础
# 使用神经网络进行分类
X, y = make_classification(n_samples=100, n_redundant=0, n_features=3, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率为：{accuracy:.2f}")

# 5.集成学习方法
# 使用梯度提升机进行分类
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率为：{accuracy:.2f}")
