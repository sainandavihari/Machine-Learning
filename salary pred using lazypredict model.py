import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pip install lazypredict
import lazypredict
salary=pd.read_csv(r'/Users/sainandaviharim/Downloads/3rd, 4th, - svr, dtr, rf,knn/emp_sal.csv')
salary.head()

x=salary.iloc[:,1:2].values
y=salary.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)


from lazypredict.Supervised import LazyRegressor
reg=LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)