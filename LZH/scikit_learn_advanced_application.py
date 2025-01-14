"""
sklearn库实操：
高级应用
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
import numpy as np

# 1.文本数据处理
texts = ["I love programming in Python", "Python is a great language", "I enjoy learning new things"]
labels = [1, 1, 0]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 2.时间序列分析
time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
values = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
X = time_series.reshape(-1, 1)
y = values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 3.图像识别基础
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 4.推荐系统
ratings = np.array([[5, 3, 0, 1],
                    [4, 0, 0, 1],
                    [1, 1, 0, 5],
                    [1, 0, 0, 4],
                    [0, 1, 5, 4]])
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(ratings)
reconstructed_ratings = np.dot(X_svd, svd.components_)
print("Reconstructed Rating:\n", reconstructed_ratings)
