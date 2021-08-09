#Jiaqi
#20210808

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('./CP1.csv', usecols=[3,5,6,9])
#drop all the 0 row
data= data.drop(data[data['cop']<0.01].index)
data= data.drop(data[data['cop']>15].index)
X_train = data[['coolingLoad','returnTemp','supplyTemp']].values[:200]
y_train = data['cop'].values[:200]


df = pd.read_csv('all_chiller.csv',usecols=[1,2,9,18])
#drop all the 0 row
df= df.drop(df[df['cop']<0.01].index)
df= df.drop(df[df['cop']>15].index)

X_test =df[['冷水机组冷冻供水温度', '冷水机组冷冻回水温度', '冷水机组负荷比']].values
y_test = df['cop'].values

#update
X_train = np.append(X_train,X_test[:100],axis=0)
y_train = np.append(y_train,y_test[:100],axis=0)
X_test = X_test[100:]
y_test = y_test[100:]
#
# ss_X=StandardScaler()
# ss_y=StandardScaler()
#
# X_train=ss_X.fit_transform(X_train)
# X_test=ss_X.transform(X_test)
# y_train=ss_y.fit_transform(y_train.reshape(-1, 1))
# y_test=ss_y.transform(y_test.reshape(-1, 1))

poly_svr=SVR(kernel='poly')   #多项式核函数初始化的SVR
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

print(' ')
print('The mean squared error of Poly SVR is',mean_squared_error(y_test,
                                                                 poly_svr_y_predict))
MAE = mean_absolute_error(y_test,poly_svr_y_predict)
print('The mean absolute error of Poly SVR is',MAE)


loss_list = []
for i in range(y_test.__len__()):
    loss = y_test[i]-poly_svr_y_predict[i]
    loss_list.append(loss)

plt.scatter(range(len(loss_list)),loss_list)
plt.title('MAE:'+str(MAE))
plt.show()

