import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection Dashboard")

API_URL = "http://127.0.0.1:8000"

# Single Transaction Prediction
st.header("Single Prediction")

with st.form("single_prediction"):
    col1, col2 = st.columns(2)

    amount = col1.number_input("Transaction Amount", value=1000.0)
    category = col1.selectbox("Category", ["food", "electronics", "clothing", "travel", "utility"])
    location = col1.selectbox("Location", ["urban", "suburban", "rural"])
    device = col1.selectbox("Device", ["mobile", "desktop", "tablet"])

    is_international = col2.selectbox("Is International?", [0, 1])
    is_weekend = col2.selectbox("Is Weekend?", [0, 1])
    hour = col2.slider("Transaction Hour", 0, 23, 13)
    is_late_night = int(hour in [0, 1, 2, 3, 4])
    is_business_hours = int(9 <= hour <= 17)

    risk_device = int(device == "desktop")
    risk_location = int(location == "urban")
    high_amount_flag = int(amount > 1000)

    amount_zscore = round((amount - 1000) / 750, 2)
    amount_deviation_flag = int(abs(amount_zscore) > 2)

    user_transaction_freq = col2.number_input("User Transaction Frequency", min_value=1, value=5)
    merchant_transaction_freq = col2.number_input("Merchant Transaction Frequency", min_value=1, value=10)

    risk_score = (
        2 * high_amount_flag +
        2 * is_international +
        risk_device +
        is_late_night +
        amount_deviation_flag
    )

    submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "amount": amount,
            "category": category,
            "location": location,
            "device": device,
            "is_international": is_international,
            "is_weekend": is_weekend,
            "hour": hour,
            "is_late_night": is_late_night,
            "is_business_hours": is_business_hours,
            "risk_device": risk_device,
            "risk_location": risk_location,
            "high_amount_flag": high_amount_flag,
            "amount_zscore": amount_zscore,
            "amount_deviation_flag": amount_deviation_flag,
            "user_transaction_freq": user_transaction_freq,
            "merchant_transaction_freq": merchant_transaction_freq,
            "risk_score": risk_score
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            if "prediction" in result:
                st.success(f"Prediction: {result['prediction']}")
                st.info(f"Probability of fraud: {result['probability']}")

                if "shap_values" in result:
                    st.subheader("Feature Contributions")
                    shap_df = pd.DataFrame.from_dict(result["shap_values"], orient="index")
                    shap_df["SHAP Value"] = shap_df.apply(
                        lambda row: row[0] if isinstance(row[0], (int, float)) else row[0][0],
                        axis=1
                    )
                    shap_df = shap_df[["SHAP Value"]]
                    shap_df = shap_df.sort_values(by="SHAP Value", key=abs)

                    fig, ax = plt.subplots(figsize=(7, min(0.4 * len(shap_df), 10)))
                    shap_df.plot.barh(
                        ax=ax,
                        color=["green" if v < 0 else "red" for v in shap_df["SHAP Value"]]
                    )
                    ax.set_title("SHAP Values")
                    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                    ax.set_xlabel("Impact")
                    st.pyplot(fig)
            else:
                st.warning("Unexpected response format.")
                st.json(result)

        except Exception as e:
            st.error(f"Request failed: {e}")

# Batch Prediction
st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        res = requests.post(f"{API_URL}/predict-batch", files={"file": uploaded_file})
        result = res.json()

        if isinstance(result, dict) and "predictions" in result:
            df_result = pd.DataFrame(result["predictions"])
            st.success("Predictions generated.")
            st.dataframe(df_result)

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

            if "shap_summary" in result and isinstance(result["shap_summary"], dict):
                st.subheader("Batch Feature Contributions")
                shap_df = pd.DataFrame.from_dict(result["shap_summary"], orient="index")
                shap_df["SHAP Value"] = shap_df.apply(
                    lambda row: row[0] if isinstance(row[0], (int, float)) else row[0][0],
                    axis=1
                )
                shap_df = shap_df[["SHAP Value"]]
                shap_df = shap_df.sort_values(by="SHAP Value", key=abs)

                fig, ax = plt.subplots(figsize=(7, min(0.4 * len(shap_df), 10)))
                shap_df.plot.barh(
                    ax=ax,
                    color=["green" if v < 0 else "red" for v in shap_df["SHAP Value"]]
                )
                ax.set_title("Average SHAP Values")
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                ax.set_xlabel("Impact")
                st.pyplot(fig)
            else:
                st.info("SHAP summary not provided in response.")

        else:
            st.warning("Unexpected response structure.")
            st.json(result)

    except Exception as e:
        st.error(f"Upload failed: {e}")