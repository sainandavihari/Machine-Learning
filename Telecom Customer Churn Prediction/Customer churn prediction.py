import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

dataset=pd.read_csv(r'/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Churn Prediction/Churn_Modelling.csv')
dataset.head()
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# converting categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])


# apply one hot encodeing to geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

# FeaTURE Scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

# split the data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# Building ANN
ann=tf.keras.models.Sequential()

# building 2 hidden layers
# here dense is used to connect layer and nuron
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

# building output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# training the ANN
ann.compile(optimizer='adamax',loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(x_train,y_train,batch_size=32,epochs=100)

# predicting the test results
y_pred=ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# with epoch=20 , optimizer=adam, acc=85.6
# with epoc=100,optimizer=adam , acc=86

# with epoc=100 , optimizer='rmsprop',acc=86.1
# with epoc=100 , optimizer='adagrad',acc=80

# best acc we got is optimizer='rmsprop' and epocs=100
# with epoc=100 , optimizer='adamax',acc=85.75