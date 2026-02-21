import streamlit as st
import requests

st.title("Neuroforge ML Prediction")

uploaded_file = st.file_uploader("Upload CSV for Prediction", type="csv")

if uploaded_file:
    st.write("Sending file to ML API...")
    response = requests.post("http://127.0.0.1:8001/predict", files={"file": uploaded_file})
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Error connecting to ML API")