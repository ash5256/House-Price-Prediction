import streamlit as st
import numpy as np
import pickle

# Load the trained model
#model = pickle.load(open('https://raw.githubusercontent.com/ash5256/House-Price-Prediction/blob/main/new_rf_model.pkl', 'rb'))
import requests
import joblib
 
model_url='https://raw.githubusercontent.com/ash5256/House-Price-Prediction/blob/main/new_rf_model.pkl'
r=requests.get(model_url)
 
if r.status_code==200:
    with open('new_rf_model.pkl','wb') as f:
        f.write(r.content)
else:
    print("Failed to download the model file")
model = joblib.load('new_rf_model.pkl')
# Streamlit app
st.title('House Price Prediction')

# Input widgets for features
area = st.number_input('Area')
bedrooms = st.number_input('Bedrooms')
bathrooms = st.number_input('Bathrooms')
stories = st.number_input('Stories')
mainroad = st.number_input('Main Road (0 for No, 1 for Yes)')
#guestroom = st.number_input('Guest Room (0 for No, 1 for Yes)')
guestroom = st.selectbox("Guest room (Yes:1 , No:0)", [1, 0])
basement = st.number_input('Basement (0 for No, 1 for Yes)')
hotwaterheating = st.number_input('Hot Water Heating (0 for No, 1 for Yes)')
airconditioning = st.number_input('Air Conditioning (0 for No, 1 for Yes)')
parking = st.number_input('Parking')
prefarea = st.number_input('Preferred Area (0 for No, 1 for Yes)')
furnishingstatus = st.number_input('Furnishing Status (0 for No, 1 for Yes)')
semifurnishingstatus = st.number_input('SemiFurnishing Status (0 for No, 1 for Yes)')

# Button to trigger prediction
if st.button('Predict Price'):
    # Create a numpy array with the input data
    data = np.array([area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                     hotwaterheating, airconditioning, parking, prefarea, furnishingstatus, semifurnishingstatus]).reshape(1, -1)

    # Make prediction using the model
    result = model.predict(data) * 1000000

    # Display the result
    st.success(f'Predicted price: {result[0]}')
