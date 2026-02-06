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
# Load ML Model & Scaler
# ===============================
xgb_model = joblib.load("best_xgboost_model.joblib")
scaler = joblib.load("scaler.joblib")   # 🔥 REQUIRED FIX

# ===============================
# App Title
# ===============================
st.title("📈 Stock Closing Price Prediction")
st.markdown(
    "This app predicts the **closing stock price** using a trained **XGBoost model** "
    "and provides **AI-based explanations** using **Gemini API**."
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

    # Prepare input
    input_data = np.array([[open_price, high_price, low_price, price_range, avg_price]])

    # 🔥 SCALE INPUT (CRITICAL FIX)
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = xgb_model.predict(input_data_scaled)[0]

    st.success(f"💰 Predicted Closing Price: ₹ {prediction:.2f}")

    # ===============================
    # AI Explanation
    # ===============================
    st.subheader("🤖 AI Explanation")

    prompt = f"""
You are a financial data science assistant.

The predicted stock closing price is ₹{prediction:.2f}.

IMPORTANT CONTEXT:
- Prediction comes from an XGBoost regression model
- The model uses ONLY numerical price inputs
- Inputs:
  Open = {open_price}
  High = {high_price}
  Low  = {low_price}

TASK:
Explain the prediction in simple terms for a beginner investor.

RULES:
- Do NOT mention news, earnings, volume, institutions, Nifty, Sensex
- Explain using price range behavior only
- Clearly say this is a statistical estimate
- Keep it short and professional
"""

    try:
        response = gemini_model.generate_content(prompt)
        st.info(response.text)

    except Exception:
        st.info(
            f"""
**Model-Based Explanation**

The predicted closing price of ₹{prediction:.2f} is derived from how the stock
moved between its daily high and low.

• Wider price range indicates higher volatility  
• The model compares this pattern with historical data  
• The output is a **statistical estimate**, not financial advice  

⚠️ External factors like news or sentiment are not included.
"""
        )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, XGBoost & Gemini API")
