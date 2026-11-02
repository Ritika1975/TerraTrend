
import joblib

# Load Linear Regression model
linear_model = joblib.load("terratrend_linear_model.pkl")

# Load XGBoost model
xgb_model = joblib.load("terratrend_xgboost_model.pkl")

print("Linear Regression model loaded successfully!")
print("XGBoost model loaded successfully!")
