import streamlit as st
import pandas as pd
import joblib

st.title('Mall Customer Segmentation Project')

# Function to reset user inputs
def reset_inputs():
    st.experimental_rerun()

# Function to make predictions
def predict_cluster(user_data):
    reconstructed_model = joblib.load('gmm_pipeline.joblib')
    pred = reconstructed_model.predict(user_data)
    return pred

# Get user inputs
st.sidebar.header('User Input')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age', 18, 100, 25)
annual_income = st.sidebar.slider('Annual Income (k$)', 0, 200, 50)
spending_score = st.sidebar.slider('Spending Score (1-100)', 1, 100, 50)

if st.sidebar.button("Reset"):
    reset_inputs()

if st.sidebar.button('Predict'):
    user_data = pd.DataFrame({'Gender': [gender], 'Age': [age], 'Annual Income (k$)': [annual_income], 'Spending Score (1-100)': [spending_score]})
    predicted_cluster = predict_cluster(user_data)
    st.subheader('Predicted Customer Cluster is:')
    st.write(predicted_cluster)

# Show user inputs in the main section
st.write('User Inputs:')
st.write(f'Gender: {gender}')
st.write(f'Age: {age}')
st.write(f'Annual Income (k$): {annual_income}')
st.write(f'Spending Score (1-100): {spending_score}')
