# -------------------------------
# Required Imports
# -------------------------------
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Sample Dataset (replace later)
# -------------------------------
data = {
    "Open": [100, 102, 101, 105, 107, 110, 108, 112, 115, 117],
    "High": [105, 106, 104, 108, 110, 113, 111, 115, 118, 120],
    "Low":  [98, 100, 99, 103, 105, 108, 106, 110, 113, 115],
    "Close":[104, 105, 103, 107, 109, 112, 110, 114, 117, 119]
}

df = pd.DataFrame(data)

# -------------------------------
# Feature Engineering
# -------------------------------
df["Price_Range"] = df["High"] - df["Low"]
df["Avg_Price"] = (df["High"] + df["Low"]) / 2

X = df[["Open", "High", "Low", "Price_Range", "Avg_Price"]]
y = df["Close"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Hyperparameter Space
# -------------------------------
param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# -------------------------------
# RandomizedSearchCV
# -------------------------------
rand_xgb = RandomizedSearchCV(
    estimator=XGBRegressor(
        random_state=42,
        objective="reg:squarederror"
    ),
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
rand_xgb.fit(X_train_scaled, y_train)

# -------------------------------
# Best Model Evaluation
# -------------------------------
best_xgb = rand_xgb.best_estimator_

y_pred = best_xgb.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# -------------------------------
# 🔥 SAVE MODEL + SCALER
# -------------------------------
joblib.dump(best_xgb, "best_xgboost_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ best_xgboost_model.joblib saved")
print("✅ scaler.joblib saved")
