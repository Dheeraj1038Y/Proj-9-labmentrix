import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1E2329;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #2D333B;
}
.metric-title {
    font-size: 18px;
    color: #AAAAAA;
}
.metric-value {
    font-size: 34px;
    font-weight: bold;
    color: white;
}
.sidebar .sidebar-content {
    background-color: #161B22;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(" Tesla Stock Predictor")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Project Information

**Model:** GRU Deep Learning

**Lookback Window:** 30 Days

**Features:** 13

**Prediction Horizon:** Next Day Closing Price

**Framework:** TensorFlow + Streamlit

**Dataset:** TSLA Engineered Features
""")

st.sidebar.markdown("---")

# =====================================================
# LOAD PACKAGE
# =====================================================

@st.cache_resource
def load_package():
    package = joblib.load("deployment_package_complete.joblib")
    return package

package = load_package()

model = package["model"]
target_scaler = package["scalers"]["target_scaler"]
feature_columns = package["feature_names"]

# =====================================================
# LOAD DATA & REGENERATE SCALER
# =====================================================

# 1. Load the historical engineered dataset
df = pd.read_csv("TSLA_engineered_features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# 2. Fix the missing feature_scaler by regenerating it on the dataset
feature_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaler.fit(df[feature_columns])

# =====================================================
# CREATE INPUT SEQUENCE
# =====================================================

feature_data = df[feature_columns]

# Get the last 30 days of data
last_30_days = feature_data.tail(30)

# 3. Scale the input features using the regenerated feature_scaler
last_30_days_scaled = feature_scaler.transform(last_30_days)

# Reshape the SCALED data into the 3D sequence expected by the GRU (1 sequence, 30 days, 13 features)
X_input = np.array(last_30_days_scaled).reshape(1, 30, 13)

# =====================================================
# PREDICTION
# =====================================================

# Make the prediction
prediction_scaled = model.predict(X_input, verbose=0)

# Inverse transform to get the real dollar amount
predicted_price = target_scaler.inverse_transform(prediction_scaled)[0][0]

# Get the current (most recent) price
current_price = df["Close"].iloc[-1]

# Calculate changes
price_change = predicted_price - current_price
percent_change = (price_change / current_price) * 100

# =====================================================
# HEADER
# =====================================================

st.title("🚗 Tesla Stock Price Prediction Dashboard")

st.markdown(
    "Deep Learning Forecast using GRU Neural Network"
)

st.markdown("---")

# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

# FIX: Format the delta string so the '-' sign is the very first character!
if price_change < 0:
    delta_str = f"-${abs(price_change):,.2f}"
else:
    delta_str = f"+${price_change:,.2f}"

with col1:
    st.metric(
        "Current Price",
        f"${current_price:,.2f}"
    )

with col2:
    st.metric(
        "Predicted Next Day Price",
        f"${predicted_price:,.2f}",
        delta=delta_str
    )

with col3:
    st.metric(
        "Expected Change",
        f"{percent_change:+.2f}%"
    )

# =====================================================
# CHART DATA & PLOTLY CHART
# =====================================================

recent_df = df.tail(30).copy()
future_date = recent_df["Date"].max() + pd.Timedelta(days=1)

fig = go.Figure()

# Plot historical actuals
fig.add_trace(
    go.Scatter(
        x=recent_df["Date"],
        y=recent_df["Close"],
        mode="lines",
        name="Historical Close Price",
        line=dict(color='#45B7D1', width=2)
    )
)

# Plot predicted line
fig.add_trace(
    go.Scatter(
        x=[recent_df["Date"].iloc[-1], future_date],
        y=[current_price, predicted_price],
        mode="lines+markers",
        name="Predicted Next Day",
        line=dict(dash='dash', color='#FF6B6B' if percent_change < 0 else '#96CEB4', width=2),
        marker=dict(size=10)
    )
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    title="Tesla Stock Price 30-Day Context & Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# DATA PREVIEW
# =====================================================

st.markdown("---")

st.subheader("Latest Historical Data")

st.dataframe(
    df[['Date', 'Close'] + feature_columns].tail(10),
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Tesla Stock Prediction Internship Capstone | Deep Learning GRU Model"
)