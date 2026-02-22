# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

st.title("Self-Driving ML Predictor")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column (What to predict)", df.columns)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    # Detect problem type
    if y.dtype in ['int64', 'float64'] and len(y.unique()) > 20:
        problem_type = "regression"
        st.info("Problem detected: **Regression**")
    else:
        problem_type = "classification"
        st.info("Problem detected: **Classification**")

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Select models
    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42)
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model_name = None
    best_score = -np.inf
    best_model_pipeline = None
    results = {}

    st.subheader("Training and Evaluating Models...")

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if problem_type == "regression":
            score = r2_score(y_test, y_pred)
            results[name] = round(score, 4)
        else:
            score = accuracy_score(y_test, y_pred)
            results[name] = round(score, 4)

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model_pipeline = pipeline

    st.subheader("Model Evaluation Results:")
    st.table(pd.DataFrame.from_dict(results, orient='index', columns=["Score"]))

    st.success(f"Best Model: **{best_model_name}** with Score = {round(best_score,4)}")

    st.subheader("Predictions on Uploaded Dataset:")
    predictions = best_model_pipeline.predict(X)
    output_df = df.copy()
    output_df["Prediction"] = predictions
    st.dataframe(output_df.head())

    # Download button
    csv = output_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )




