"""
sklearn库实操：
无监督学习
"""

import os

os.environ["OMP_NUM_THREADS"] = '2'
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import pandas as pd

# 1.聚类分析
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
kmeans = KMeans(n_clusters=4, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')
plt.show()

# 2.主成分分析
iris = load_iris()
X = iris.data
y = iris.target
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=50, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 3.奇异值分解
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, s=50, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 4.关联规则学习
data = {'Milk': [1, 0, 1, 1, 0],
        'Bread': [0, 1, 1, 1, 1],
        'Butter': [1, 1, 0, 1, 1],
        'Bear': [0, 0, 1, 0, 1]}
df = pd.DataFrame(data)
# 使用Apriori算法发现频繁项集
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
# 生成关联规则
rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric="confidence",
                          min_threshold=0.7)
print(rules)
