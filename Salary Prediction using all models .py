import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salary=pd.read_csv(r'/Users/sainandaviharim/Downloads/3rd, 4th, - svr, dtr, rf,knn/emp_sal.csv')
salary.head()

x=salary.iloc[:,1:2].values
y=salary.iloc[:,2].values


# applying linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
regressor_predict=regressor.predict([[6.5]])
print(regressor_predict)
    

# Applying SVR Algoriuthm
from sklearn.svm import SVR
svr_reg=SVR()
svr_reg.fit(x, y)

svr_model_pred=svr_reg.predict([[6.5]])
print(svr_model_pred)


# applying polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

# Model training
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Plotting
plt.scatter(x, y, color="red")
plt.plot(x_poly, lin_reg_2.predict(poly_reg.transform(x_poly[:,4])), color='blue')
plt.show()

# hyper perameter tuning of SVR
svr_reg=SVR(kernel='poly',degree=3,gamma='auto',C=1.0)
svr_reg.fit(x, y)

svr_model_pred=svr_reg.predict([[6.5]])
print(svr_model_pred)

# knn regressor
from sklearn.neighbors import KNeighborsRegressor
knn_reg= KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='ball_tree')
knn_reg.fit(x,y)

knn_pred=knn_reg.predict([[6.5]])
print(knn_pred)# output is 182.500

# decision tree regression
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(criterion="absolute_error",splitter= "random")
dt_model.fit(x,y)

dt_pred=dt_model.predict([[6.5]])
dt_pred# output is 150k without hyper perameter tuning 
# with hyper perameter tuning , o/p is 20k

# rf regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(random_state=0)# hyper perameter tuning
rf_reg.fit(x,y)
rf_predict=rf_reg.predict([[6.5]])
print(rf_predict) # output is 158.k




