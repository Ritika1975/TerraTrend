

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="TerraTrend",
    page_icon="🌍",
    layout="wide"
)


st.markdown("""
# 🌍 TerraTrend
### Smart Climate AI

Explore historical global temperature anomalies
and machine-learning-based climate forecasts.

Our system uses historical climate data and
machine learning models to analyze global warming trends.
""")


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TerraTrend | Climate Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ================= PREMIUM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #0b1d2b, #102b2d);
    color: #e8f5ef;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #e8fff5 !important;
}
p, label {
    color: #b7d5ca !important;
}
div[data-testid="stMetric"] {
    background: rgba(18, 45, 54, 0.75);
    border: 1px solid rgba(80, 220, 170, 0.25);
    padding: 20px;
    border-radius: 16px;
}
div[data-testid="stMetricValue"] {
    color: #55e6ad !important;
}
.stDownloadButton button {
    background: #22c997;
    color: #061b16;
    border-radius: 12px;
    font-weight: bold;
}
hr {
    border-color: rgba(80, 220, 170, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_resource
def load_model():
    return joblib.load("terratrend_linear_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("terratrend_historical_data.csv")

@st.cache_data
def load_results():
    return pd.read_csv("terratrend_model_results.csv")

@st.cache_data
def load_predictions():
    return pd.read_csv("terratrend_predictions.csv")

@st.cache_data
def load_future():
    return pd.read_csv("terratrend_future_predictions.csv")

model = load_model()
df = load_data()
model_results = load_results()
prediction_df = load_predictions()
future_df = load_future()

df = df.sort_values("Year").reset_index(drop=True)

last_year = int(df["Year"].max())
latest_anomaly = float(df["Monthly_Anomaly"].iloc[-1])
average_anomaly = float(df["Monthly_Anomaly"].mean())

# ================= HERO =================
st.markdown("""
# 🌍 TerraTrend
### CLIMATE INTELLIGENCE STUDIO

**Understand the past. Explore the future.**

AI-powered global land temperature anomaly forecasting.
""")

st.markdown("---")

# ================= CLIMATE SNAPSHOT =================
st.markdown("### 🌿 Climate Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Historical Records", f"{len(df):,}")
c2.metric("Coverage Begins", str(int(df["Year"].min())))
c3.metric("Latest Year", str(last_year))
c4.metric("Latest Anomaly", f"{latest_anomaly:.2f} °C")

st.markdown("---")

# ================= HISTORICAL GRAPH =================
st.markdown("## 📈 Earth's Climate Timeline")
st.caption("Annual global land temperature anomalies")

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#0b1d2b")
ax.set_facecolor("#0b1d2b")

ax.plot(
    df["Year"], df["Monthly_Anomaly"],
    color="#44e0a5", linewidth=2
)

ax.fill_between(
    df["Year"], df["Monthly_Anomaly"],
    color="#44e0a5", alpha=0.12
)

ax.axhline(0, color="white", alpha=0.3, linestyle="--")
ax.set_xlabel("Year", color="white")
ax.set_ylabel("Temperature Anomaly (°C)", color="white")
ax.tick_params(colors="white")
ax.grid(alpha=0.15)

for spine in ax.spines.values():
    spine.set_color("#24434a")

st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ================= LINEAR FORECAST =================
st.markdown("## 🔮 AI Forecast Studio")
st.caption("Explore future global land temperature anomalies")

forecast_year = st.slider(
    "Choose a future year",
    min_value=last_year + 1,
    max_value=2100,
    value=min(last_year + 5, 2100)
)

prediction = float(
    model.predict(pd.DataFrame({"Year": [forecast_year]}))[0]
)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🤖 AI Prediction")
    st.metric(
        f"Predicted anomaly in {forecast_year}",
        f"{prediction:.2f} °C"
    )
    st.info(
        "This is a global land temperature anomaly, "
        "not a city-specific temperature."
    )

with col2:
    future_years = list(range(last_year + 1, forecast_year + 1))

    future_predictions = model.predict(
        pd.DataFrame({"Year": future_years})
    )

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Predicted_Anomaly": future_predictions
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        forecast_df["Year"],
        forecast_df["Predicted_Anomaly"],
        color="#53c9ff", marker="o", linewidth=2
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly (°C)")
    ax.set_title("Linear Regression Forecast")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ================= CLIMATE INSIGHTS =================
st.markdown("## 🧠 Climate Insights")

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown("### 📊 Historical Average")
    st.write(f"{average_anomaly:.2f} °C")
    st.caption("Average anomaly across historical records.")

with i2:
    st.markdown("### 🌡️ Latest Observation")
    st.write(f"{latest_anomaly:.2f} °C")
    st.caption(f"Latest available year: {last_year}")

with i3:
    st.markdown("### 🔭 Forecast Horizon")
    st.write(f"{forecast_year - last_year} years")
    st.caption("Selected prediction distance.")

st.markdown("---")

# ================= MODEL COMPARISON =================
st.markdown("## 📊 Model Performance Comparison")

st.dataframe(model_results, width="stretch")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### MAE Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        model_results["Model"],
        model_results["MAE"],
        color=["#38bdf8", "#a78bfa", "#34d399"]
    )
    ax.set_ylabel("MAE")
    ax.set_title("Lower is Better")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### RMSE Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        model_results["Model"],
        model_results["RMSE"],
        color=["#38bdf8", "#a78bfa", "#34d399"]
    )
    ax.set_ylabel("RMSE")
    ax.set_title("Lower is Better")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

best_model = model_results.loc[
    model_results["MAE"].idxmin(), "Model"
]
st.success(f"🏆 Best Performing Model: {best_model}")

st.markdown("---")

# ================= PREDICTION ACCURACY =================
st.markdown("## 🎯 Prediction Accuracy")
st.caption("Comparison on held-out test years")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    prediction_df["Year"], prediction_df["Actual"],
    marker="o", label="Actual"
)
ax.plot(
    prediction_df["Year"], prediction_df["Linear Regression"],
    marker="s", label="Linear Regression"
)
ax.plot(
    prediction_df["Year"], prediction_df["XGBoost"],
    marker="^", label="XGBoost"
)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.set_title("Actual vs Predicted")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ================= FUTURE FORECAST COMPARISON =================
st.markdown("## 🔮 Future Forecast Comparison")
st.caption("Forecasts exported from Google Colab")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    future_df["Year"],
    future_df["Linear Regression"],
    marker="o", label="Linear Regression"
)
ax.plot(
    future_df["Year"],
    future_df["XGBoost"],
    marker="s", label="XGBoost"
)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.set_title("Future Temperature Anomaly Forecast")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.dataframe(future_df, width="stretch")

# ================= DOWNLOAD CENTER =================
st.markdown("---")
st.markdown("## 📥 Download Center")

st.download_button(
    "Download Selected Forecast",
    data=forecast_df.to_csv(index=False),
    file_name="TerraTrend_Forecast.csv",
    mime="text/csv"
)

st.download_button(
    "Download Future Model Comparison",
    data=future_df.to_csv(index=False),
    file_name="TerraTrend_Future_Comparison.csv",
    mime="text/csv"
)

st.markdown("""
---
<div style="text-align:center;color:#7fa99a;">
TerraTrend • Climate Intelligence Studio • AI for a greener future 🌍
</div>
""", unsafe_allow_html=True)


# ================= DATASET INFORMATION =================

st.markdown("---")
st.header("📊 Dataset Information")

st.write("""
TerraTrend uses historical global temperature data
from Berkeley Earth.

The dataset contains temperature observations
covering multiple centuries.

Our project uses annual average temperature anomalies
to analyze global climate trends and generate forecasts.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Historical Years", f"{df['Year'].nunique():,}")

with col2:
    st.metric("Dataset Records", f"{len(df):,}")

with col3:
    st.metric("Starting Year", f"{int(df['Year'].min())}")

# ================= MODEL EXPLANATION =================

st.markdown("---")
st.header("🤖 Machine Learning Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Linear Regression")
    st.write("""
    Linear Regression predicts future temperature
    anomalies by learning the relationship between
    year and temperature anomaly.

    It provides a simple baseline trend forecast.
    """)

with col2:
    st.subheader("🌳 XGBoost")
    st.write("""
    XGBoost is a machine learning algorithm based
    on decision trees.

    It uses historical lag values and rolling
    averages to learn patterns in temperature data.
    """)
    
st.info("""
TerraTrend compares multiple forecasting approaches
to understand their performance on historical climate data.
""")

# ================= LIMITATIONS & FUTURE SCOPE =================

st.markdown("---")
st.header("🔍 Limitations & Future Scope")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Limitations")
    st.write("""
    - Forecasts are based on historical temperature patterns.
    - The system predicts global temperature anomalies,
      not city-specific temperatures.
    - Climate anomalies depend on many factors
      beyond the models used.
    - Long-term forecasts contain uncertainty.
    """)

with col2:
    st.subheader("🚀 Future Scope")
    st.write("""
    - Integrate more climate variables.
    - Add interactive regional climate analysis.
    - Improve forecasting using advanced time-series models.
    - Include prediction uncertainty visualization.
    - Develop a real-time climate monitoring system.
    """)
