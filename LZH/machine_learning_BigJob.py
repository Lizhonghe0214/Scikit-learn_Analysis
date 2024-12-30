"""
测井自动分层
"""
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 载入训练数据集
data = pd.read_excel("datasets/logging_data.xlsx")
print(data.head())
# 分割特征和标签
X = data[["GR/API", "DT/us/m", "DEN/g/cc"]]
y = data[["层号"]]
print(X.head())
print(y.head())
# 手动确定分层点
xlim = [329.25, 344.625, 353.125, 359.625, 365.25, 371.875, 384.25, 399.5, 405.5, 414.25, 425.25, 432.5
    , 451.5, 465.5, 480.375, 488.25, 498.125, 513.125, 524.875, 535.375, 541.5, 550.625, 560.125
    , 565.5, 573.875, 585.5, 592.125, 598.125, 612.25, 627.625, 634.5, 649.75, 670.375, 687.25]

# 可视化测井分层
plt.figure(figsize=(15, 8))
plt.plot(data["DEPTH/m"], X["GR/API"], color="red")
plt.plot(data["DEPTH/m"], 0.15 * X["DT/us/m"], color="blue")
plt.plot(data["DEPTH/m"], 60 * X["DEN/g/cc"], color="black")
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(data["DEPTH/m"], X["GR/API"], color="red")
plt.plot(data["DEPTH/m"], 0.15 * X["DT/us/m"], color="blue")
plt.plot(data["DEPTH/m"], 60 * X["DEN/g/cc"], color="black")
for item in xlim:
    plt.vlines(item, 50, 140, colors="green", linestyles="--")
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(data["DEPTH/m"], X["GR/API"], color="red")
plt.plot(data["DEPTH/m"], 0.15 * X["DT/us/m"], color="blue")
plt.plot(data["DEPTH/m"], 60 * X["DEN/g/cc"], color="black")
for item in xlim:
    plt.vlines(item, 50, 140, colors="green", linestyles="--")


# 训练数据GR的预处理
num = 0  # 每层样本数量
sum = 0.0  # 每层GR总值
mean = 0.0  # 每层GR均值
temp_sum = 0.0  # 尖峰点GR总值
temp_mean = 0.0  # 尖峰点GR均值
flag = 0  # 前一个尖峰点下标
count = 0  # 每层厚度控制
a = 1  # 预处理阶梯状拔高参数

for i in range(1, 3085):
    num += 1
    count += 1
    mean += X["GR/API"][i]
    mean = sum / num
    temp_sum = X["GR/API"][i - 1] + X["GR/API"][i] + X["GR/API"][i + 1]
    temp_mean = temp_sum / 3
    if abs(mean - temp_mean) >= 8 and count >= 60:  # 8 60
        flag = int(i - num)
        for j in range(flag, i):
            X.loc[j, "GR/API"] += 100 * a  # 100
        num = 0
        count = 0
        a += 1
        plt.vlines(data["DEPTH/m"][i], 50, 140, colors="black", linestyles="--")
    elif 3085 - i <= 120:
        for j in range(i, 3087):
            X.loc[j, "GR/API"] += 100 * a  # 100
        break

plt.show()

# 特殊处理
for i in range(2940, 2970):
    if X.loc[i, "GR/API"] < 500:
        X.loc[i, "GR/API"] += 100 * (a - 1)  # 100

# GR数据处理后可视化
plt.figure(figsize=(15, 8))
plt.plot(data['DEPTH/m'], X['GR/API'], color="red")
for item in xlim:
    plt.vlines(item, 50, 5500, colors="black", linestyles="--")
plt.show()
print(X.sample(10))

"""------------------------------------------------------------"""
# 数据归一化
X_norm = X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(X_norm.head())

# 分割数据集与验证集
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
print(X_train.head())
# 训练
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("验证集模型准确率：", accuracy)

# 载入测试集
data2 = pd.read_excel("datasets/logging_test_data.xlsx")
print(data2.head())
X2 = data2[["GR/API", "DT/us/m", "DEN/g/cc"]]
y2 = data2[["层号"]]
print(X2.head())
print(y2.head())

# 测试数据预处理
num = 0
sum = 0.0
temp_sum = 0.0
mean = 0.0
temp_mean = 0.0
flag = 0
count = 0
a = 1

for i in range(1, 3772):
    num = num + 1
    count = count + 1
    sum += X2["GR/API"][i]
    mean = sum / num
    temp_sum = X2["GR/API"][i - 1] + X2["GR/API"][i] + X2["GR/API"][i + 1]
    temp_mean = temp_sum / 3
    if abs(mean - temp_mean) >= 8 and count >= 80:  # 8 80
        flag = int(i - num)
        for j in range(flag, i):
            X2.loc[j, "GR/API"] += 100 * a  # 100
        num = 0.0
        count = 0
        a += 1
    elif 3772 - i <= 150:
        for j in range(i, 3773):
            X2.loc[j, "GR/API"] += 100 * a  # 100
        break

# 特殊处理
for i in range(3550, 3622):  # 3550 3622
    if X2.loc[i, "GR/API"] < 500:
        X2.loc[i, "GR/API"] += 100 * (a - 1)  # 100

# 测试数据处理后可视化
xlim2 = [318.8, 332.6, 340, 346.3, 351.6, 357.1, 370, 384.9, 390.2, 398.8, 408.6, 415.2, 434.5, 447.9
    , 463.2, 471.3, 480.9, 495, 508, 519, 524.65, 534.3, 542.3, 548, 558, 568, 574.1, 581.1, 594.7, 611
    , 617.4, 632.9, 653.2, 669.7]
plt.figure(figsize=(15, 8))
plt.plot(data2['DEPTH/m'][1:3500], X2['GR/API'][1:3500], color="blue")
for item in xlim2:
    plt.vlines(item, 50, 5500, colors="black", linestyles="--")
plt.show()

X2_norm = X2.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(X2_norm.head())

y2_pred = clf.predict(X2_norm)
accuracy2 = accuracy_score(y2, y2_pred)
print("测试集模型准确率：", accuracy2)

# plt.figure(figsize=(30,10))
# plt.plot(data["DEPTH/m"],X["GR/API"],color="red")
# plt.plot(data2["DEPTH/m"],X2["GR/API"],color="blue")
# plt.show()
