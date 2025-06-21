# Fraud Detection System 
![Python](https://img.shields.io/badge/python-3.10-blue)
![Dockerized](https://img.shields.io/badge/containerized-Docker-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-green)
![Streamlit](https://img.shields.io/badge/ui-Streamlit-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
A complete end-to-end fraud detection pipeline that simulates transactional data, engineers meaningful features, trains multiple machine learning models with SMOTE balancing, and deploys real-time prediction APIs using **FastAPI** and an interactive **Streamlit** dashboard. SHAP is integrated for interpretability, enabling transparency into each prediction.

# Tech Stack
- Python 3.10
- FastAPI + Streamlit for backend and UI
- scikit-learn, XGBoost, Random Forest, Logistic Regression for modeling
- SMOTE for class imbalance, SHAP for interpretability
- Joblib for model serialization
- Pandas, NumPy, Matplotlib, Seaborn for data handling & visualization

# Project Structure
Fraud_Detection/ 
├── app.py                        # Streamlit frontend 
├── main.py                       # FastAPI backend 
├── fraud_detection_pipeline.py   # Data simulation + model training 
├── fraud_model.joblib 
├── raw_fraud_data.csv            # Simulated dataset 
├── requirements.txt 
├── Dockerfile                    #Containerization Setup
├── .dockerignore                 #Docker exclusions
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


## 🧪 Local Setup

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

## 🐳 Run with Docker

You can containerize and run the entire project with a single Docker command.
Note: This app runs locally over http:// — make sure not to use https:// or the browser may block the connection.

1. Build the Docker image
docker build -t fraud-detector .

2. Run the container
docker run -p 8000:8000 -p 8501:8501 fraud-detector

• 	FastAPI API Docs → http://localhost:8000/docs
• 	Streamlit Dashboard → http://localhost:8501

For batch predictions, upload sample_test.csv in the Streamlit dashboard.

# License
This project is licensed under the MIT License.



