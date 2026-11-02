)

st.markdown("""
---
<div style="text-align:center;color:#7fa99a;">
TerraTrend • Climate Intelligence Studio • AI for a greener future 🌍
</div>
""", unsafe_allow_html=True)

# ================= MODEL COMPARISON =================

st.markdown("---")
st.markdown("## 📊 Model Performance Comparison")

model_results = pd.read_csv("terratrend_model_results.csv")

st.dataframe(model_results, use_container_width=True)

# MAE Comparison
st.markdown("### MAE Comparison")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(model_results["Model"], model_results["MAE"], color=["#38bdf8", "#a78bfa", "#34d399"])
ax.set_ylabel("MAE")
ax.set_title("Model MAE (Lower is Better)")
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)

# RMSE Comparison
st.markdown("### RMSE Comparison")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(model_results["Model"], model_results["RMSE"], color=["#38bdf8", "#a78bfa", "#34d399"])
ax.set_ylabel("RMSE")
ax.set_title("Model RMSE (Lower is Better)")
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)

# Best Model
best_model = model_results.loc[model_results["MAE"].idxmin(), "Model"]

st.success(f"🏆 Best Performing Model: {best_model}")

# ================= PREDICTION ACCURACY =================

st.markdown("---")
st.markdown("## 🎯 Prediction Accuracy")

prediction_df = pd.read_csv("terratrend_predictions.csv")

st.markdown("### Actual vs Predicted Temperature Anomalies")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    prediction_df["Year"],
    prediction_df["Actual"],
    marker="o",
    label="Actual"
)

ax.plot(
    prediction_df["Year"],
    prediction_df["Linear Regression"],
    marker="s",
    label="Linear Regression"
)

ax.plot(
    prediction_df["Year"],
    prediction_df["XGBoost"],
    marker="^",
    label="XGBoost"
)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.set_title("Actual vs Predicted Global Temperature Anomalies")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# Future Forecast Comparison

last_year = int(annual_df["Year"].max())

future_years = np.arange(last_year + 1, last_year + 11)

# Linear Regression Forecast
linear_future_predictions = linear_model.predict(
    pd.DataFrame({"Year": future_years})
)

# XGBoost Future Forecast (Recursive)
history = list(annual_df["Monthly_Anomaly"].values)

xgb_future_predictions = []

for year in future_years:
    lag1 = history[-1]
    lag2 = history[-2]
    lag3 = history[-3]
    rolling_mean_3 = np.mean(history[-3:])

    features = pd.DataFrame({
        "Lag1": [lag1],
        "Lag2": [lag2],
        "Lag3": [lag3],
        "RollingMean3": [rolling_mean_3]
    })

    prediction = xgb_model.predict(features)[0]

    xgb_future_predictions.append(prediction)
    history.append(prediction)

# ================= FUTURE FORECAST COMPARISON =================

st.markdown("---")
st.markdown("## 🔮 Future Forecast Comparison")

future_df = pd.read_csv("terratrend_future_predictions.csv")

st.markdown("### Next 10 Years: Model Forecasts")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    future_df["Year"],
    future_df["Linear Regression"],
    marker="o",
    label="Linear Regression"
)

ax.plot(
    future_df["Year"],
    future_df["XGBoost"],
    marker="s",
    label="XGBoost"
)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")