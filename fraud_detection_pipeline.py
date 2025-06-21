import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import shap
import matplotlib.pyplot as plt

# Set working directory (for Colab/Drive, adjust as needed)
project_dir = '/content/drive/MyDrive/copilot_FraudDetection2'
os.makedirs(project_dir, exist_ok=True)
os.chdir(project_dir)

# ------------------------
# Simulate transaction data
# ------------------------
def simulate_transaction_data(n_samples=10000, fraud_ratio=0.05, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    def random_date():
        return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

    def gen(n, fraud=False):
        return pd.DataFrame({
            'user_id': np.random.randint(1000, 5000, n),
            'merchant_id': np.random.randint(200, 3000, n),
            'amount': np.abs(np.round(np.random.normal(5000 if fraud else 1000,
                                                       3000 if fraud else 750, n), 2)),
            'category': np.random.choice(['food', 'electronics', 'clothing', 'travel', 'utility'], n),
            'location': np.random.choice(['urban', 'suburban', 'rural'], n),
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], n),
            'is_international': np.random.choice([0, 1], size=n, p=[0.3, 0.7] if fraud else [0.95, 0.05]),
            'is_weekend': np.random.randint(0, 2, size=n),
            'timestamp': [random_date() for _ in range(n)],
            'label': int(fraud)
        })

    df = pd.concat([gen(n_legit), gen(n_fraud, fraud=True)]).sample(frac=1).reset_index(drop=True)
    return df

# ------------------------
# Feature engineering
# ------------------------
def add_risk_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['is_late_night'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['risk_device'] = df['device'].isin(['desktop', 'public_terminal']).astype(int)
    df['risk_location'] = df['location'].isin(['urban']).astype(int)
    df['high_amount_flag'] = (df['amount'] > 1000).astype(int)
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['amount_deviation_flag'] = (df['amount_zscore'].abs() > 2).astype(int)
    df['user_transaction_freq'] = df.groupby('user_id')['user_id'].transform('count')
    df['merchant_transaction_freq'] = df.groupby('merchant_id')['merchant_id'].transform('count')
    df['risk_score'] = (
        2 * df['high_amount_flag'] +
        2 * df['is_international'] +
        df['risk_device'] +
        df['is_late_night'] +
        df['amount_deviation_flag']
    )
    return df

# ------------------------
# Preprocessing
# ------------------------
def preprocess_for_modeling(df):
    df = df.copy()
    df.drop(['user_id', 'merchant_id', 'timestamp'], axis=1, inplace=True)
    encoders = {}
    for col in ['category', 'location', 'device']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    X = df.drop('label', axis=1)
    y = df['label']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    return X_train, X_test, y_train, y_test, encoders

# ------------------------
# Model training and evaluation
# ------------------------
def benchmark_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False,
                                 eval_metric='logloss', random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, preds, output_dict=True)
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score'],
            'ROC AUC': auc
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, digits=4))
        print(f"ROC AUC: {auc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    return results

def auto_select_model(metrics_dict, by_metric='F1-Score'):
    return max(metrics_dict.items(), key=lambda x: x[1][by_metric])[0]

def get_model_by_name(name):
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)
    elif name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == 'XGBoost':
        return XGBClassifier(n_estimators=100, eval_metric='logloss',
                             use_label_encoder=False, random_state=42)

# ------------------------
# Main pipeline execution
# ------------------------
df = simulate_transaction_data()
df.to_csv('raw_fraud_data.csv', index=False)
df = add_risk_features(df)
X_train, X_test, y_train, y_test, encoders = preprocess_for_modeling(df)
metrics = benchmark_models(X_train, y_train, X_test, y_test)
best_model_name = auto_select_model(metrics)
final_model = get_model_by_name(best_model_name)
final_model.fit(X_train, y_train)
model_file = f"{best_model_name.replace(' ', '_').lower()}_fraud_model.joblib"
joblib.dump({'model': final_model, 'encoders': encoders}, model_file)

# ------------------------
# SHAP explainability
# ------------------------
X_sample = X_test.sample(n=50, random_state=42).reset_index(drop=True)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list) and len(shap_values) == 2:
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap.summary_plot(shap_values[:, :, 1], X_sample, plot_type="bar")
else:
    print("Unrecognized SHAP format. Skipping plot.")