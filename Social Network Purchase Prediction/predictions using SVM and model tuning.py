import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score

data= pd.read_csv(r'/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Social Networks purchase prediction using Naive bayes/Social_Network_Ads.csv')


x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

sc=StandardScaler()
x=sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

classifier=SVC()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
print(cm,acc)
# Here , the accuracy of classifier model is 95%

bias=classifier.score(x_train,y_train)
variance=classifier.score(x_test,y_test)
print(bias,variance)
# bias and variance values are 90 and 95 . So This is the best fit model , but still trying to improve accuracy by using cross validation techniques


# Applying k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=15)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
 
# Here , The accuracy value is changing or improving while changing the value of cv

# Applying grid search cv  to find best model and best perameter
from sklearn.model_selection import GridSearchCV
perameters=[{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

gridsearch_cv=GridSearchCV(estimator=classifier,param_grid=perameters,scoring='accuracy',cv=10)
gridsearch_cv.fit(x_train,y_train)
best_accuracy=gridsearch_cv.best_score_
best_perameters=gridsearch_cv.best_params_
print("best accuracy is {:.2f} %" .format(accuracies.mean()*100))
print("best perameters are :",best_perameters)


