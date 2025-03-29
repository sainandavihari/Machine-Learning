import streamlit as slt
import pickle
import numpy as np

model=pickle.load(open(r'/Users/sainandaviharim/Desktop/Python Projects/ML Projects/Salary Prediction App/linear_regression_model.pkl','rb'))

slt.title("Salary Prediction App")

slt.write("This app predicts salary based on experience using simple linear regression model")

Years_Exp=slt.number_input("Enter Year of Experience:", min_value=0.0,max_value=50.0,value=1.0,step=0.5)

if slt.button("Predict Salary"):
    exp_input=np.array([[Years_Exp]])
    prediction=model.predict(exp_input)

    slt.success(f"The predicted salary for {Years_Exp} years of experience is :${prediction[0]:.2f}")
slt.write("The model was trained using a dataset of salaries and years of experience.built model by nanda vihari")