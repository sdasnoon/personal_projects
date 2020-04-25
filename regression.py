# Data center : https://www.kaggle.com/andonians/random-linear-regression/data
# https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
import pandas as pd
import numpy as np
import os
path = os.path.dirname(os.path.abspath("__file__"))
df_train = pd.read_csv('/resources/regression_train.csv')
df_test = pd.read_csv('/resources/regression_test.csv')

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
