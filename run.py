from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# Allow cross-origin requests (useful when opening the HTML file locally or from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ أضف الجزء ده مباشرة بعد إنشاء app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ممكن تخصصها لو عايز
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("uvicorn.error")

# ✅ استبدل الـ run_id بالـ id الحقيقي من MLflow UI
RUN_ID = "e31729ad9b084dc7a34bfaf73c69585b"
 
MODEL_URI = f"runs:/{RUN_ID}/model"

# Do not load the model at import time — that makes importing fail if MLflow
# or the tracking server is unavailable. Load it on startup and handle errors.
model = None
expected_features = None
scaler = None
numeric_features = None
categorical_features = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    global expected_features, scaler, numeric_features, categorical_features
    try:
        logger.info("⏳ Loading model from MLflow...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("✅ Model loaded successfully from MLflow!")
    except Exception as e:
        # Log the error but don't crash the app import — return 503 from endpoints if model missing.
        logger.exception("Failed to load model from MLflow: %s", e)
        model = None

    # Reconstruct expected feature columns and fit a scaler so we can transform incoming requests
    try:
        df = pd.read_csv('global_cancer_patients_2015_2024.csv')
        # reproduce the same preprocessing from the notebook: numeric features + one-hot of categoricals
        numeric_features = df.select_dtypes(include=[np.int64, np.float64]).columns.tolist()
        # ensure target is not included
        if 'Target_Severity_Score' in numeric_features:
            numeric_features.remove('Target_Severity_Score')
        # remove Patient_ID if accidentally included
        if 'Patient_ID' in numeric_features:
            numeric_features.remove('Patient_ID')
        categorical_features = df.select_dtypes(include=[object]).columns.tolist()
        # remove Patient_ID from categorical as well
        if 'Patient_ID' in categorical_features:
            categorical_features.remove('Patient_ID')

        data = pd.concat([df[numeric_features], pd.get_dummies(df[categorical_features])], axis=1)
        # drop target if present
        if 'Target_Severity_Score' in data.columns:
            data = data.drop(columns=['Target_Severity_Score'])
        expected_features = data.columns.tolist()
        # fit scaler on the full data so we can standardize incoming requests (model was trained on scaled data)
        scaler = StandardScaler()
        scaler.fit(data.values)
        logger.info("✅ Reconstructed feature columns and fitted scaler: %d features", len(expected_features))
    except Exception as e:
        logger.exception("Failed to reconstruct features/scaler from CSV: %s", e)
        expected_features = None
        scaler = None


# Accept the original raw columns from the CSV (we drop Patient_ID and Target_Severity_Score).
# Assumption: you want 12 input fields for the API (Year removed). If you'd rather keep Year,
# tell me and I'll swap it.
class CancerData(BaseModel):
    Age: float
    Gender: str
    Country_Region: str
    Genetic_Risk: float
    Air_Pollution: float
    Alcohol_Use: float
    Smoking: float
    Obesity_Level: float
    Cancer_Type: str
    Cancer_Stage: str
    Treatment_Cost_USD: float
    Survival_Years: float


@app.post("/predict_severity")
def predict(data: CancerData):
    if model is None:
        # Service unavailable until the model is loaded successfully
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again later.")

    if expected_features is None:
        # can't transform inputs to expected shape
        raise HTTPException(status_code=503, detail="Model feature metadata not available. Try again later.")

    # Build a full feature vector matching training columns. Fill missing (one-hot) columns with 0.
    # Start with zeros for all expected features
    try:
        # Build a raw dataframe using the original numeric + categorical columns
        raw = data.dict()
        # Ensure we have the lists to encode
        if numeric_features is None or categorical_features is None:
            raise HTTPException(status_code=503, detail="Feature metadata not available; can't encode input.")

        # Build numeric part: if a numeric feature expected by training isn't present in API model, we set 0
        numeric_part = {col: 0 for col in numeric_features}
        for col in numeric_part.keys():
            if col in raw:
                numeric_part[col] = raw[col]

        # Build categorical part from API fields
        cat_input = {}
        for col in categorical_features:
            # if the categorical column was in the CSV but not part of our API model (e.g., Patient_ID), skip
            if col in raw:
                cat_input[col] = raw[col]

        raw_df = pd.DataFrame([numeric_part])
        if len(cat_input) > 0:
            cat_df = pd.DataFrame([cat_input])
            # One-hot encode the categorical input; align with training dummies by reindexing after concat
            cat_dummies = pd.get_dummies(cat_df)
            input_encoded = pd.concat([raw_df.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
        else:
            input_encoded = raw_df

        # Reindex to expected features (missing columns -> 0)
        input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

        # scale
        if scaler is not None:
            X_scaled = scaler.transform(input_encoded.values)
        else:
            X_scaled = input_encoded.values

        prediction = model.predict(X_scaled)[0]
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    if prediction < 3:
        category = "Low"
    elif prediction < 6:
        category = "Medium"
    else:
        category = "High"

    return {"score": float(prediction), "severity": category}
