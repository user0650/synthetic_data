"""线性回归：真实数据训练模型，与合成数据训练模型，并对比模型效果"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data = pd.read_csv("/Users/sunxiaopeng9/PycharmProjects/synthetic_data/data/train_data_2.csv", header=None)
print(data.head())
print(data.shape)

feature_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
label_col = 13

# 真实数据建模
print("Real Model:")
X = data[feature_cols]
y = data[label_col]

X_train = X[0:100000]
X_test = X[100000:]
y_train = y[0:100000]
y_test = y[100000:]

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)

y_pred = lr.predict(X_test)
print("Real Model RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# 合成数据建模
data1 = pd.read_csv("/Users/sunxiaopeng9/PycharmProjects/synthetic_data/data/train_data_2_syn.csv", header=None)
print(data1.head())
print(data1.shape)

print("Synthetic Model:")
X1 = data1[feature_cols]
y1 = data1[label_col]

X1_train = X1[0:15000]
X1_test = X1[15000:]
y1_train = y1[0:15000]
y1_test = y1[15000:]

lr1 = LinearRegression()
lr1.fit(X1_train, y1_train)
print(lr1.intercept_)
print(lr1.coef_)

y1_pred = lr1.predict(X1_test)
print("Synthetic Model RMSE:", np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
