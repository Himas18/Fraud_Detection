# Fraud Detection System 

A complete end-to-end fraud detection pipeline that simulates transactional data, engineers meaningful features, trains multiple machine learning models with SMOTE balancing, and deploys real-time prediction APIs using **FastAPI** and an interactive **Streamlit** dashboard. SHAP is integrated for interpretability, enabling transparency into each prediction.

# Tech Stack
- Python 3.10
- FastAPI + Streamlit for backend and UI
- scikit-learn, XGBoost, Random Forest, Logistic Regression for modeling
- SMOTE for class imbalance, SHAP for interpretability
- Joblib for model serialization
- Pandas, NumPy, Matplotlib, Seaborn for data handling & visualization

# Project Structure
fraud_detection/ 
├── app.py                        # Streamlit frontend 
├── main.py                       # FastAPI backend 
├── fraud_detection_pipeline.py   # Data simulation + model training 
├── random_forest_fraud_model.joblib 
├── raw_fraud_data.csv            # Simulated dataset 
├── requirements.txt 
├── README.md

#  Features

- Generate synthetic fraud transaction data with customizable ratio
- Engineer risk-aware features (e.g., late-night, international, z-score)
- Apply **SMOTE** to balance fraud/non-fraud classes
- Train and evaluate three models with benchmarking
- Auto-select best model based on F1-score
- REST endpoints for both single and batch predictions
- SHAP summary plots for feature contribution analysis
- Streamlit UI for interactive predictions and uploads


# 📈 SHAP Visualizations

SHAP is used to explain predictions on both individual and batch transactions:

- For single predictions: horizontal bar plot of most impactful features
- For batch uploads: aggregated SHAP summary


# Local Setup

1. Clone this repository

bash
git clone https://github.com/Himas18/Fraud_Detection.git
cd Fraud_Detection

2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the FastAPI backend
uvicorn main:app --reload

5. Run the Streamlit frontend
streamlit run app.py

For batch predictions, upload the sample CSV
Refer to `sample_test.csv` for the required input format in batch predictions.

# License
This project is licensed under the MIT License.



