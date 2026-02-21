from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import joblib
from automl_trainer import train_automl
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI(title="Neuroforge AutoML Service")

# Folders
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# /predict endpoint (existing)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")

        # Save uploaded CSV
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Train model automatically
        train_automl(file_path)

        # Load trained model & feature columns
        model = joblib.load(os.path.join(MODEL_FOLDER, "final_model.pkl"))
        features = joblib.load(os.path.join(MODEL_FOLDER, "feature_columns.pkl"))

        # Load CSV for prediction
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        X = pd.get_dummies(X)

        # Feature matching
        for col in features:
            if col not in X.columns:
                X[col] = 0
        X = X[features]

        # Predict
        predictions = model.predict(X)

        # Load model info
        best_score = joblib.load(os.path.join(MODEL_FOLDER, "best_score.pkl"))
        best_model = joblib.load(os.path.join(MODEL_FOLDER, "best_model_name.pkl"))
        all_scores = joblib.load(os.path.join(MODEL_FOLDER, "all_scores.pkl"))

        dataset_info = {
            "columns": list(df.columns),
            "total_columns": len(df.columns),
            "target_column": df.columns[-1]
        }

        return {
            "message": "Neuroforge Prediction Completed",
            "best_algorithm": best_model,
            "accuracy_score": float(best_score),
            "all_models_score": all_scores,
            "dataset_info": dataset_info,
            "total_rows": len(predictions),
            "predictions": predictions.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
# /train endpoint (for Spring Boot)

@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        # Unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")

        # Save CSV
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Call automl trainer
        train_automl(file_path)

        return JSONResponse(content={"status": "Training Started Successfully"})

    except Exception as e:
        print(f"Error in training: {e}")
        return JSONResponse(content={"status": "Training Failed", "error": str(e)})