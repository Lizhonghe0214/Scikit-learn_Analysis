#导入库文件
import numpy as np #科学计算库
import matplotlib.pyplot as plt #绘图库可视化函数
import pandas as pd #数据处理库，数据分析库
import seaborn as sns #高级数据可视化库

from sklearn.model_selection import train_test_split #数据分割库
#加载数据集
fruis_df = pd.read_table("C:/Users/闵子洋/Desktop/MZY/MZY/datasets/fruit_data_with_colors.txt")
print(fruis_df.head(3))
#获取数据集中的特征名称
fruis_name_dict = dict(zip(fruis_df["fruit_label"], fruis_df["fruit_name"]))
print("___________")
print(fruis_name_dict)
X = fruis_df[["mass", "width", "height", "color_score"]]
y = fruis_df["fruit_label"]
#分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
print("数据集总共：{}, 训练集：{}, 测试集合：{}".format(len(X), len(X_train), len(X_test)))
#绘制散点图
sns.pairplot(data=fruis_df, hue="fruit_name", vars=["mass", "width", "height", "color_score"])
plt.show()
#建立模型
from sklearn.neighbors import KNeighborsClassifier #KNN分类器的算法库
knn = KNeighborsClassifier(n_neighbors=5)
#训练
knn.fit(X_train, y_train)
#预测
y_pred = knn.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score #准确率
accuracy_score(y_test, y_pred) # y_test:实际值，y_pred:预测值

# 在不同k值下对应准确率的可视化模型
k_range = range(1, 21)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([i for i in range(1, 21)])
plt.show()
