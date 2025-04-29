import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,accuracy_score

data= pd.read_csv(r'/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Social Networks purchase prediction using Naive bayes/Social_Network_Ads.csv')


x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

# split the data into test data and train data
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)


#sc=StandardScaler()
#sc=Normalizer()
#x_train=sc.fit_transform(x_train)
#x_test=sc.transform(x_test)


classifier=DecisionTreeClassifier(criterion="entropy",max_depth=10,random_state=0)

classifier.fit(x_train,y_train)

# predict the future
y_pred=classifier.predict(x_test)


# perforamance measures of classifications
cm=confusion_matrix(y_test, y_pred)
print(cm)

accuracy=accuracy_score(y_test, y_pred)
print(accuracy)

bias=classifier.score(x_train,y_train)
variance=classifier.score(x_test,y_test)

print(bias,variance)


# Feature scalling techniques are not required for decision tree algorithmns.
# before feature scaling , after feature feature scalling , tghe accuracy is same.
# After hyper perameter tuning , the accuracy is changed from 90 to 92.


# Therefore , TThe above model is the best model.


