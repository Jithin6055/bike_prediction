# Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('bike_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('Bike Type Prediction')

# Input fields for the user
milage = st.number_input('Milage')
price = st.number_input('Price')
bike_weight = st.number_input('Bike Weight')

# Prediction
if st.button('Predict'):
    input_data = np.array([[milage, price, bike_weight]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.write(f'The predicted bike type is: {prediction[0]}')
