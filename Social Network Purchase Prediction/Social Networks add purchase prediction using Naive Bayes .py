import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB

from sklearn.metrics import confusion_matrix,accuracy_score

data= pd.read_csv(r'/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Social Networks purchase prediction using Naive bayes/Social_Network_Ads.csv')


x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

# split the data into test data and train data
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)


sc=StandardScaler()
#sc=Normalizer()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#classifier=BernoulliNB()
classifier=GaussianNB()
#classifier=MultinomialNB()

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

# by using BernouliNB , accuracy = 82.5 , bias=70, variance=82.5
#by using  GaussianNB , accuracy=91.25, bias=88, variance=91
# by using MultinomialNB, Accuracy =72.5, bias=62, variance=72.5


# Therefore , The best model is built by using GaussianNB Algarithm.