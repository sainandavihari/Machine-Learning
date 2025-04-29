import streamlit as st
import pickle
import numpy as np

file_name='model_final.pkl'
with open(file_name,'rb') as file:
    model_load=pickle.load(file)

st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        text-align: center;
        font-size: 32px;
    }
    .text {
        color: #7D3C98;
        text-align: center;
        font-size: 18px;
    }
    .prediction {
        color: #6C3483;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="text">Enter your height in feet to predict your weight.</p>', unsafe_allow_html=True)

default_height = 5.8

height_input=st.number_input("Enter the height in feet :",value=default_height,min_value=0.0)

if st.button('Predict'):
    #height_input_2d=np.array(height_input).reshape(1, -1)
    pred_weight=model_load.predict(height_input)
    st.markdown(f'<p class="prediction">Predicted weight: {pred_weight[0, 0]} kg</p>', unsafe_allow_html=True)
