# app.py

import streamlit as st
import pandas as pd
import joblib

st.title("ML Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read original CSV to keep original columns
    df_original = pd.read_csv(uploaded_file)
    original_columns = df_original.columns.tolist()

    # Load model and feature list
    model = joblib.load("final_model.pkl")
    features = joblib.load("feature_columns.pkl")  # list of feature columns

    # Prepare data for prediction
    df = df_original.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with 0

    X = df[features]

    # Make predictions
    predictions = model.predict(X)
    df["Prediction"] = predictions

    # If classification model, show probabilities
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        # Assuming binary classification: probability of class 1
        df["Prediction_Prob"] = prob[:, 1]

    # Show only original columns + predictions (and probabilities if classification)
    display_columns = original_columns + ["Prediction"]
    if "Prediction_Prob" in df.columns:
        display_columns.append("Prediction_Prob")

    st.subheader("Prediction Results")
    st.dataframe(df[display_columns])

    # Allow download
    csv = df[display_columns].to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )


