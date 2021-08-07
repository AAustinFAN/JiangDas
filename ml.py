import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_data = np.array(pd.read_csv('./CP1.csv'))
train_unknown_list = []

for x in range(train_data.shape[0]):
    if train_data[x][-1] == 0.0:
        train_unknown_list.append(x)

train_data = np.delete(train_data, train_unknown_list, axis=0)
X_train = train_data[:100, (6, 5, 3)]
y_train = train_data[:100, -1]

df = pd.read_csv('all_chiller.csv')
df = df.drop(labels=417)
#df = df.drop(labels=30)
test_data = df.drop(range(0, 30))
X_test = np.array(test_data[['冷水机组冷冻供水温度', '冷水机组冷冻回水温度', '冷水机组负荷比']])[:2000]
y_test = np.array(test_data['cop'])[:2000]


ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1, 1))
y_test=ss_y.transform(y_test.reshape(-1, 1))

poly_svr=SVR(kernel='linear')   #多项式核函数初始化的SVR
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

print(' ')
print('R-squared value of Poly SVR is',poly_svr.score(X_test,y_test))
print('The mean squared error of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                   ss_y.inverse_transform(poly_svr_y_predict)))
