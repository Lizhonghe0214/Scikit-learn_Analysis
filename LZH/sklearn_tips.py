"""
Scikit learn tips
based on
https://github.com/justmarkham/scikit-learn-tips
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.datasets import load_diabetes, load_wine, load_iris
from sklearn.metrics import roc_auc_score
from sklearn import set_config

# tip 1:使用ColumnTransformer对不同的列应用不同的预处理
X = pd.DataFrame({'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                  'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q'],
                  'Sex': ['male', 'female', 'female', 'female', 'male', 'male'],
                  'Age': [22, 38, 26, 35, 35, np.nan]})
print(X)
ohe = OneHotEncoder()
imp = SimpleImputer()
ct = make_column_transformer((ohe, ['Embarked', 'Sex']),
                             (imp, ['Age']),
                             remainder='passthrough')
print(ct.fit_transform(X))

# tip 2:使用ColumnTransformer的7种选择列的方法
ct = make_column_transformer((ohe, ['Embarked', 'Sex']))
ct = make_column_transformer((ohe, [1, 2]))
ct = make_column_transformer((ohe, slice(1, 3)))
ct = make_column_transformer((ohe, [False, True, True, False]))
ct = make_column_transformer((ohe, make_column_selector(pattern='E|S')))
ct = make_column_transformer((ohe, make_column_selector(dtype_include=object)))
ct = make_column_transformer((ohe, make_column_selector(dtype_exclude='number')))
print(ct.fit_transform(X))

# tip 3:两种编码类别特征的常用方法
X1 = pd.DataFrame({'Shape': ['square', 'square', 'oval', 'circle'],
                   'Class': ['third', 'first', 'second', 'third'],
                   'Size': ['S', 'S', 'L', 'XL']})
print(X1)
# OneHotEncoder用于无序数据
ohe = OneHotEncoder(sparse_output=False)
print(ohe.fit_transform(X1[['Shape']]))
# OrdinalEncoder用于有序数据
oe = OrdinalEncoder(categories=[['first', 'second', 'third'], ['S', 'M', 'L', 'XL']])
print(oe.fit_transform(X1[['Class', 'Size']]))

# tip 4:设置handle_unknown='ignore'将新类别编码为全零
X2 = pd.DataFrame({'col': ['A', 'B', 'C', 'B']})
X2_new = pd.DataFrame({'col': ['A', 'C', 'D']})
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
print(X2)
print(ohe.fit_transform(X2[['col']]))
print(X2_new)
print(ohe.transform(X2_new[['col']]))

# tip 5:pipeline:将多个步骤链接在一起：每个步骤的输出用作下一步的输入
train = pd.DataFrame({'feat1': [10, 20, np.nan, 2],
                      'feat2': [25., 20, 5, 3],
                      'label': ['A', 'A', 'B', 'B']})
test = pd.DataFrame({'feat1': [30., 5, 15],
                     'feat2': [12, 10, np.nan]})
clf = LogisticRegression()
# 两步管道：填充缺失值，然后将结果传递给分类器
pipe = make_pipeline(imp, clf)
print(train)
print(test)
features = ['feat1', 'feat2']
X3, y3 = train[features], train['label']
X3_new = test[features]
pipe.fit(X3, y3)
print(pipe.predict(X3_new))

# tip 6:估算缺失值时，可以保留那些值缺失的信息，并将其作为特征
X4 = pd.DataFrame({'Age': [20, 30, 10, np.nan, 10]})
print(X4)
imputer = SimpleImputer(add_indicator=True)
print(imputer.fit_transform(X4))

# tip 7:random_state确保“随机”过程每次都会输出相同的结果
X5 = X[['Fare', 'Embarked', 'Sex']]
y5 = X[['Age']]
print(X5)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.5, random_state=1)
print(X5_train)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.5, random_state=2)
print(X5_train)

# tip 8:填补缺失值的更好方法
X6 = pd.DataFrame({'SibSp': [1, 1, 0, 1, 0, 0],
                   'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                   'Age': [22, 38, 26, 35, 35, np.nan]})
print(X6)
imputer_it = IterativeImputer()
print(imputer_it.fit_transform(X6))
imputer_knn = KNNImputer(n_neighbors=2)
print(imputer_knn.fit_transform(X6))

# tip 9:Pipeline需要对步骤进行命名，而make_pipeline不需要
X7 = X[['Embarked', 'Sex', 'Age', 'Fare']]
ct = make_column_transformer((ohe, ['Embarked', 'Sex']),
                             (imp, ['Age']),
                             remainder='passthrough')
pipe = make_pipeline(ct, clf)
ct = ColumnTransformer([('encoder', ohe, ['Embarked', 'Sex']),
                        ('imputer', imp, ['Age'])],
                       remainder='passthrough')
pipe = Pipeline([('preprocessor', ct),
                 ('classifier', clf)])

# tip 10:named_steps检查管道的中间步骤
df = pd.DataFrame({'Age': [22, 38, 26, 35, 35, np.nan],
                   'Pclass': [3, 1, 3, 1, 3, 3],
                   'Survived': [0, 1, 1, 1, 0, 0]})
print(df)
X8 = df[['Age', 'Pclass']]
y8 = df['Survived']
pipe = make_pipeline(imp, clf)
pipe.fit(X8, y8)
print(pipe.named_steps.simpleimputer.statistics_)
print(pipe.named_steps.logisticregression.coef_)

# tip 11:显示线性模型的截距和系数
dataset = load_diabetes()
X9, y9 = dataset.data, dataset.target
features = dataset.feature_names
model = LinearRegression()
model.fit(X9, y9)
print(model.intercept_)
print(model.coef_)
print(list(zip(features, model.coef_)))

# tip 12:设置stratify=y，以便在分割时保持类的比例
df = pd.DataFrame({'feature': list(range(8)),
                   'target': ['not fraud'] * 6 + ['fraud'] * 2})
X10 = df[['feature']]
y10 = df['target']
X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y10, test_size=0.5, random_state=0)
print(y10_train)
print(y10_test)
X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y10, test_size=0.5, random_state=0, stratify=y10)
print(y10_train)
print(y10_test)

# tip 11:对分类特征的缺失值进行插补
X11 = pd.DataFrame({'Shape': ['square', 'square', 'oval', 'circle', np.nan]})
print(X11)
imputer = SimpleImputer(strategy='most_frequent')
print(imputer.fit_transform(X11))
imputer = SimpleImputer(strategy='constant', fill_value='missing')
print(imputer.fit_transform(X11))

# tip 12:使用KFold或StratifiedKFold来洗牌
X_reg, y_reg = load_diabetes(return_X_y=True)
reg = LinearRegression()
kf = KFold(5, shuffle=True, random_state=1)
print(cross_val_score(reg, X_reg, y_reg, cv=kf, scoring='r2'))

# tip 13:评估指标AUC
X12, y12 = load_wine(return_X_y=True)
X12 = X12[:, 0:2]
clf = LogisticRegression()
X12_train, X12_test, y12_train, y12_test = train_test_split(X12, y12, random_state=0)
clf.fit(X12_train, y12_train)
y12_score = clf.predict_proba(X12_test)
# 基于训练集/测试集划分的多分类AUC
print(roc_auc_score(y12_test, y12_score, multi_class='ovo'))
# 基于交叉验证的多类AUC
print(cross_val_score(clf, X12, y12, cv=5, scoring='roc_auc_ovo').mean())

# tip 14:在ColumnTransformer或管道中进行特征工程
X13 = pd.DataFrame({'Fare': [200, 300, 50, 900],
                    'Code': ['X12', 'Y20', 'Z7', np.nan],
                    'Deck': ['A101', 'C102', 'A200', 'C300']})
# 将现有函数转换为transformer
clip_values = FunctionTransformer(np.clip, kw_args={'a_min': 100, 'a_max': 600})


# 将自定义函数转换为transformer
def first_letter(df):
    return df.apply(lambda x: x.str.slice(0, 1))


get_first_letter = FunctionTransformer(first_letter)
ct = make_column_transformer((clip_values, ['Fare']),
                             (get_first_letter, ['Code', 'Deck']))
print(X13)
print(ct.fit_transform(X13))

# tip 15:加载数据集
df = load_iris(as_frame=True)['frame']
print(df.head(3))
X14, y14 = load_iris(as_frame=True, return_X_y=True)
print(X14.head(3))
print(y14.head(3))

# tip 16:参数管理
clf = LogisticRegression(C=0.1, solver='liblinear')
print(clf)
print(clf.get_params())
set_config(print_changed_only=False)
print(clf)

# tip 17:二元特征
X15 = pd.DataFrame({'Shape': ['circle', 'oval', 'square', 'square'],
                    'Color': ['pink', 'yellow', 'pink', 'yellow']})
print(X15)
# drop=None（默认）为每个类别创建一个特性列
ohe = OneHotEncoder(sparse_output=False, drop=None)
print(ohe.fit_transform(X15))
# drop='first'删除每个特征中的第一个类别
ohe = OneHotEncoder(sparse_output=False, drop='first')
print(ohe.fit_transform(X15))
# drop='if_binary'删除第一个二元特征的类别
ohe = OneHotEncoder(sparse_output=False, drop='if_binary')
print(ohe.fit_transform(X15))

# tip 18:通过某些列并删除其他列
imputer = SimpleImputer()
X16 = pd.DataFrame({'A': [1, 2, np.nan],
                    'B': [10, 20, 30],
                    'C': [100, 200, 300],
                    'D': [1000, 2000, 3000],
                    'E': [10000, 20000, 30000]})
print(X16)
ct = make_column_transformer((imputer, ['A']),
                             ('passthrough', ['B', 'C']),
                             remainder='drop')
print(ct.fit_transform(X16))
ct = make_column_transformer((imputer, ['A']),
                             ('drop', ['D', 'E']),
                             remainder='passthrough')
print(ct.fit_transform(X16))

# tip 19:特征交互
X17 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [4, 4, 4],
                    'C': [0, 10, 100]})
poly = PolynomialFeatures(include_bias=False, interaction_only=True)
print(X17)
print(poly.fit_transform(X17))
