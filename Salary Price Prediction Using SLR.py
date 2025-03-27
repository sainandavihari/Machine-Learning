import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =pd.read_csv(r'/Users/sainandaviharim/Downloads/Salary_Data.csv')
dataset.head()

x=dataset.iloc[:,0]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=20,random_state=0)


x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
    
y_pred=regressor.predict(x_test)


# plotting the best fit line
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


# predicting codeff and slope

coef=regressor.coef_
intercept=regressor.intercept_


# predicting 12 years of exp person salary 
exp_12_years= coef*12+intercept

exp_12_years


# checking bias and variance
bias= regressor.score(x_train, y_train)
variance=regressor.score(x_test,y_test)

print(bias,variance)

# here , bias and variance values are 0.94 and 0.95 so , this is the good model for the dataset 


# Applying statistics to the data frame

print(dataset.mean())
print(dataset.median())
print(dataset.std())
print(dataset.mode())
print(dataset.var())

from scipy.stats import variation,stats
variation(dataset.values)# this will give coefficient of variation
variation(dataset['Salary'])

print(dataset.corr())
print(dataset['Salary'].corr(dataset['YearsExperience']))

# skewness - measures of variability
print(dataset.skew())
print(dataset['Salary'].skew())

# standared error 

dataset.sem()# standared error for entire dataframe
dataset['Salary'].sem() # standared error of the salary atttribute

# applying standardization to the dataset /  feature scalling
dataset.apply(stats.zscore)
# apply feature scalling to the Salary attribute
stats.zscore(dataset['Salary'])

# finding degree of freedom
a=dataset.shape[0]
b=dataset.shape[1]

degree_of_freedom=a-b
print(degree_of_freedom)


# SSR (Sum of square Regressor)
y_mean= np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE (Sum of Square of Error)
y=y[0:6]
SSE=np.sum((y-y_mean))

SST=SSR+SSE


R_Squared= 1-(SSR/SST)

print(R_Squared)






