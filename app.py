import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai

# ===============================
# Gemini API Setup (ONCE)
# ===============================
genai.configure(api_key="AIzaSyBIFD9H_0Dskw5NQg-EsOdnSIiM8Eo6VSc")
gemini_model = genai.GenerativeModel("models/gemini-flash-latest")

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="centered"
)

# ===============================
# Load ML Model (NO SCALER)
# ===============================
xgb_model = joblib.load("best_xgboost_model.joblib")

# ===============================
# App Title
# ===============================
st.title("📈 Stock Closing Price Prediction")
st.markdown(
    "This app predicts the **closing stock price** using a trained **XGBoost model** "
    "and provides **AI-based explanations**."
)

# ===============================
# User Inputs
# ===============================
st.subheader("🔢 Enter Stock Details")

open_price = st.number_input("Open Price", min_value=0.0, step=1.0)
high_price = st.number_input("High Price", min_value=0.0, step=1.0)
low_price = st.number_input("Low Price", min_value=0.0, step=1.0)

# Feature Engineering
price_range = high_price - low_price
avg_price = (high_price + low_price) / 2

# ===============================
# Prediction
# ===============================
if st.button("📊 Predict Closing Price"):

    input_data = np.array([
        [open_price, high_price, low_price, price_range, avg_price]
    ])

    # ✅ DIRECT PREDICTION (NO SCALER)
    prediction = xgb_model.predict(input_data)[0]

    st.success(f"💰 Predicted Closing Price: ₹ {prediction:.2f}")

    # ===============================
    # AI Explanation
    # ===============================
    st.subheader("🤖 AI Explanation")

    try:
        response = gemini_model.generate_content(
            f"""
Explain the predicted closing price of ₹{prediction:.2f}
using only price-based reasoning.

Inputs:
Open = {open_price}
High = {high_price}
Low = {low_price}

Rules:
- No news, earnings, sentiment, volume
- Explain using price range behavior only
- Mention this is a statistical estimate
- Keep it short and beginner-friendly
"""
        )
        st.info(response.text)

    except Exception:
        st.info(
            f"""
**Model-Based Explanation**

The predicted closing price of ₹{prediction:.2f} is calculated using
today’s price movement.

• The model evaluates how far price moved between high and low  
• It compares this pattern with historical data  
• The result is a statistical estimate, not financial advice  

⚠️ External factors like news or sentiment are not included.
"""
        )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, XGBoost & Gemini API")
