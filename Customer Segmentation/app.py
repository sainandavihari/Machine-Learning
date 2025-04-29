import streamlit as slt
import pickle
import numpy as np

model=pickle.load(open(r'/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Customer Segmentation/kmeans1.pkl','rb'))

slt.title('Customer Segmentation App')

slt.write("This app is based on Annual income(k$) and Spending score(0-100) then this app can give a rating on customer purchase(whether he/she can purtchase or can't purchase) ")

Annual_income=slt.number_input("Enter Annual Income(k$):", min_value=0,max_value=100,value=1,step=1)

Spending_score=slt.number_input("Enter spending score(0-100):", min_value=0,max_value=100,value=1,step=1)

if slt.button("Predict rating"):
    Input1=np.array([[Annual_income,Spending_score]])

    prediction=model.predict(Input1)

    slt.success(f"The predicted rating is :{prediction[0]}")
slt.write("The model was trained using a dataset of customers in a mall .built model by nanda vihari")