from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import io

app = FastAPI(title="Fraud Detection API", debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model_package = joblib.load("fraud_model.joblib")
model = model_package["model"]
encoders = model_package["encoders"]
expected_columns = list(model.feature_names_in_)

# SHAP explainer
explainer = shap.TreeExplainer(model)

categorical_cols = ["category", "location", "device"]
numeric_cols = [
    "amount", "is_international", "is_weekend", "hour",
    "is_late_night", "is_business_hours", "risk_device", "risk_location",
    "high_amount_flag", "amount_zscore", "amount_deviation_flag",
    "user_transaction_freq", "merchant_transaction_freq", "risk_score"
]
feature_cols = numeric_cols + categorical_cols

class Transaction(BaseModel):
    amount: float
    category: str
    location: str
    device: str
    is_international: int
    is_weekend: int
    hour: int
    is_late_night: int
    is_business_hours: int
    risk_device: int
    risk_location: int
    high_amount_flag: int
    amount_zscore: float
    amount_deviation_flag: int
    user_transaction_freq: int
    merchant_transaction_freq: int
    risk_score: int

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])

    for col in categorical_cols:
        if col in encoders:
            try:
                data[col] = encoders[col].transform([data[col][0]])
            except ValueError:
                return {"error": f"Unknown category '{data[col][0]}' in column '{col}'"}

    try:
        data = data[expected_columns]
        prob = model.predict_proba(data)[0][1]
        pred = int(prob >= 0.5)

        shap_vals = explainer.shap_values(data)
        shap_values = shap_vals[1][0] if isinstance(shap_vals, list) and len(shap_vals) > 1 else shap_vals[0]
        shap_dict = dict(zip(data.columns, shap_values.tolist()))

        return {
            "prediction": "Fraud" if pred == 1 else "Legit",
            "probability": round(prob, 4),
            "shap_values": shap_dict
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception:
        return {"error": "Invalid or unreadable file"}

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return {"error": f"Missing required columns: {missing}"}

    try:
        input_df = df[expected_columns].copy()
    except KeyError as e:
        return {"error": f"Column mismatch: {str(e)}"}

    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    input_df.dropna(subset=numeric_cols, inplace=True)

    for col in categorical_cols:
        if col in encoders:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError as e:
                return {"error": f"Invalid categories in '{col}': {str(e)}"}

    try:
        probas = model.predict_proba(input_df)[:, 1]
        preds = (probas >= 0.5).astype(int)

        output_df = input_df.copy()
        output_df["prediction"] = ["Fraud" if p else "Legit" for p in preds]
        output_df["probability"] = probas.round(4)

        shap_vals = explainer.shap_values(input_df)
        shap_matrix = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else shap_vals
        mean_shap = shap_matrix.mean(axis=0)
        shap_summary = dict(zip(expected_columns, mean_shap.tolist()))

        return {
            "predictions": output_df[["prediction", "probability"]].to_dict(orient="records"),
            "shap_summary": shap_summary
        }

    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}