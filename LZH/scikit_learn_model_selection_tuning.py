"""
sklearn库实操：
模型选择与调优
"""

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# 1.交叉验证
iris = load_iris()
X = iris.data
y = iris.target
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# 2.超参数优化
model = SVC()
# 定义超参数搜索空间
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 3.模型选择策略
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
models = [LogisticRegression(max_iter=200),
          RandomForestClassifier(n_estimators=100),
          SVC()]
for item in models:
    item.fit(X_train, y_train)
    y_pred = item.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{item.__class__.__name__} accuracy:{accuracy}")

# 4.特征选择
selector = SelectKBest(f_classif, k=2)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train_new, y_train)
y_pred = model.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after feature selection:", accuracy)
