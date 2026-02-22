import streamlit as st
import pandas as pd
import joblib

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    model = joblib.load("final_model.pkl")
    features = joblib.load("feature_model.pkl")  # list of feature columns

    df = pd.read_csv(uploaded_file)
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    df["Prediction"] = model.predict(X)

    st.dataframe(df)
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")

