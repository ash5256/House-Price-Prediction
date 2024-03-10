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
area = area = st.number_input("Area (in square feet)",min_value=1000,max_value=17000)
bedrooms = st.number_input("bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("bathrooms", min_value=1, max_value=10, step=1)
stories = st.number_input("Stories", min_value=1, max_value=10, step=1)
mainroad = st.selectbox("Main Road (Yes:1 , No:0)", [1, 0])
#guestroom = st.number_input('Guest Room (0 for No, 1 for Yes)')
guestroom = st.selectbox("Guest room (Yes:1 , No:0)", [1, 0])
basement = st.selectbox("basement(Yes:1 , No:0)", [1, 0])
hotwaterheating = st.selectbox("hot water heating(Yes:1 , No:0)", [1, 0])
airconditioning = st.selectbox("Air Conditioning(Yes:1 , No:0)", [1, 0])
parking = st.number_input("Parking", min_value=0,max_value=4,step=1) 
prefarea = st.selectbox("Prefered Area(Yes:1 , No:0)", [1, 0])
furnishingstatus = st.selectbox("Furnishing Status(Yes:1 , No:0)", [1, 0])
semifurnishingstatus = st.selectbox("SemiFurnishing Status(Yes:1 , No:0)", [1, 0])

# Button to trigger prediction
if st.button('Predict Price'):
    # Create a numpy array with the input data
    data = np.array([area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                     hotwaterheating, airconditioning, parking, prefarea, furnishingstatus, semifurnishingstatus]).reshape(1, -1)

    # Make prediction using the model
    result = model.predict(data) * 1000000

    # Display the result
    st.success(f'Predicted price: {result[0]}')
