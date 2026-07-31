import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="UK Weather Forecasting Dashboard",
    page_icon="🌧",
    layout="wide"
)

st.title("UK Weather Forecasting Dashboard")
st.markdown(
    """
    Welcome to the dissertation dashboard for **UK Temperature & Precipitation Forecasting**.
    Use the sidebar to navigate between pages:

    - **Model Comparison** — RMSE / MAE / R2 across all models (RF, XGBoost, ARIMA, SARIMA, Holt-Winters, Transformer, Prophet, Persistence)
    - **Seasonal Analysis** — RMSE broken down by season (Winter, Spring, Summer, Autumn)
    - **Regional Analysis** — Predictions and errors broken down by UK region
    - **Extreme Events** — Hybrid ensemble performance on extreme rainfall days
    """
)

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/test_predictions_dummy.csv", parse_dates=["date"])
    return df

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total records", f"{len(df):,}")
col2.metric("Date range", f"{df['date'].min().date()} to {df['date'].max().date()}")
col3.metric("Regions covered", df["region"].nunique())

st.subheader("Preview of predictions data")
st.dataframe(df.head(20), use_container_width=True)
