# Data center : https://www.kaggle.com/andonians/random-linear-regression/data
# https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a

# Using Sklearn Library
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath("__file__"))
df_train = pd.read_csv(path+'/resources/regression_train.csv')
df_test = pd.read_csv(path+'/resources/regression_test.csv')
df_train = df_train.dropna()
df_test = df_test.dropna()

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)



clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print('Coefficients: \n', clf.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test,y_pred))


print(r2_score(y_test,y_pred))


plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.legend()
plt.show()
