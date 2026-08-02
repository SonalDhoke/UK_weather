import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UK Weather Forecasting Dashboard",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #F7F9FC;
    border-right: 1px solid #E5E9F0;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 10px 24px 10px;
}
.sidebar-logo span.icon {
    font-size: 30px;
}

/* Radio group acts as nav list */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: transparent;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 2px;
    font-size: 15px;
    font-weight: 500;
    color: #2E3A4B;
    width: 100%;
    transition: background-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #EEF2F8;
    cursor: pointer;
}
/* Active/selected radio row */
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #E7F0FF !important;
    color: #1A56DB !important;
    font-weight: 700 !important;
}
/* Hide the actual radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
    display: none;
}

.sidebar-footer {
    position: fixed;
    bottom: 20px;
    padding: 0 14px;
    font-size: 12px;
    color: #9AA5B5;
    line-height: 1.5;
}

/* KPI card style tweaks */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E9F0;
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data
def load_predictions():
    return pd.read_csv("Dataset/test_predictions_dummy.csv", parse_dates=["date"])

@st.cache_data
def load_model_comparison():
    return pd.read_csv("Dataset/model_comparison_results.csv")

@st.cache_data
def load_seasonal():
    return pd.read_csv("Dataset/seasonal_rmse.csv")

@st.cache_data
def load_intensity():
    return pd.read_csv("Dataset/intensity_rmse.csv")

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION (radio button logic)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo"><span class="icon">🌧</span></div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        label="Navigation",
        options=[
            "🏠  Dashboard",
            "📊  Model Comparison",
            "🌦  Seasonal Analysis",
            "📍  Regional Breakdown",
            "⚠️  Extreme Events",
            "ℹ️  About"
        ],
        label_visibility="collapsed",
        key="nav"
    )

    st.markdown(
        """
        <div class="sidebar-footer">
        UK Precipitation Flood-Risk<br>Forecast Dashboard<br>&copy; 2026
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("UK Weather Forecasting Dashboard")
    st.markdown(
        """
        Welcome to the dissertation dashboard for **UK Temperature & Precipitation Forecasting**.
        Use the sidebar to navigate between pages:

        - **Model Comparison** — RMSE / MAE / R2 across all models (RF, XGBoost, ARIMA, SARIMA, Holt-Winters, Transformer, Prophet, Persistence)
        - **Seasonal Analysis** — RMSE broken down by season (Winter, Spring, Summer, Autumn)
        - **Regional Breakdown** — Predictions and errors broken down by UK region
        - **Extreme Events** — Hybrid ensemble performance on extreme rainfall days
        """
    )

    df = load_predictions()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", f"{len(df):,}")
    col2.metric("Date range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    col3.metric("Regions covered", df["region"].nunique())

    st.subheader("Preview of predictions data")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: MODEL COMPARISON
# ---------------------------------------------------------------------------
def render_model_comparison():
    st.title("Model Comparison")

    df = load_model_comparison()
    target = st.selectbox("Select target variable", df["Target"].unique())
    sub = df[df["Target"] == target].sort_values("RMSE")

    st.dataframe(sub, use_container_width=True)

    fig = px.bar(
        sub, x="Model", y="RMSE", color="Model",
        title=f"RMSE by Model ({target})",
        text_auto=".3f"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        sub, x="Model", y="R2", color="Model",
        title=f"R-squared by Model ({target})",
        text_auto=".3f"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: SEASONAL ANALYSIS
# ---------------------------------------------------------------------------
def render_seasonal_analysis():
    st.title("Seasonal Analysis")

    df = load_seasonal()
    st.dataframe(df, use_container_width=True)

    melted = df.melt(
        id_vars=["season", "n_samples"],
        value_vars=["RMSE_XGBoost", "RMSE_Hybrid"],
        var_name="Model", value_name="RMSE"
    )
    fig = px.bar(
        melted, x="season", y="RMSE", color="Model", barmode="group",
        title="Seasonal RMSE: XGBoost vs Hybrid Ensemble",
        text_auto=".3f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rainfall Intensity Breakdown")
    idf = load_intensity()
    st.dataframe(idf, use_container_width=True)

    imelt = idf.melt(
        id_vars=["intensity", "n_samples"],
        value_vars=["RMSE_XGBoost", "RMSE_Hybrid"],
        var_name="Model", value_name="RMSE"
    )
    fig2 = px.bar(
        imelt, x="intensity", y="RMSE", color="Model", barmode="group",
        title="RMSE by Rainfall Intensity Bucket",
        text_auto=".3f"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: REGIONAL BREAKDOWN
# ---------------------------------------------------------------------------
def render_regional_breakdown():
    st.title("Regional Breakdown")

    df = load_predictions()
    regions = st.multiselect(
        "Select region(s)", df["region"].unique(),
        default=list(df["region"].unique())
    )
    filtered = df[df["region"].isin(regions)]

    grouped = filtered.groupby("region").agg(
        mean_precip_true=("precip_true", "mean"),
        mean_precip_pred_xgb=("pred_precip_xgb", "mean"),
        mean_precip_pred_hybrid=("pred_precip_hybrid", "mean"),
        n_records=("region", "count")
    ).reset_index()

    st.dataframe(grouped, use_container_width=True)

    fig = px.bar(
        grouped.melt(
            id_vars="region",
            value_vars=["mean_precip_true", "mean_precip_pred_xgb", "mean_precip_pred_hybrid"],
            var_name="Series", value_name="Precipitation (mm)"
        ),
        x="region", y="Precipitation (mm)", color="Series", barmode="group",
        title="Average Precipitation: Actual vs Predicted by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Time series for selected region(s)")
    if regions:
        region_choice = st.selectbox("Zoom into one region", regions)
        ts = filtered[filtered["region"] == region_choice].sort_values("date")

        fig2 = px.line(
            ts, x="date", y=["precip_true", "pred_precip_xgb", "pred_precip_hybrid"],
            title=f"Precipitation over time — {region_choice}"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: EXTREME EVENTS
# ---------------------------------------------------------------------------
def render_extreme_events():
    st.title("Extreme Rainfall Events")

    df = load_predictions()
    threshold = df["extreme_threshold"].iloc[0]

    st.metric("Extreme threshold (90th percentile)", f"{threshold:.2f} mm")

    actual_extreme = df["is_extreme_actual"].sum()
    pred_extreme = df["is_extreme_pred_hybrid"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Actual extreme days", int(actual_extreme))
    col2.metric("Predicted extreme days (Hybrid)", int(pred_extreme))

    extreme_df = df[df["is_extreme_actual"] == 1]
    st.subheader("Actual extreme rainfall events")
    st.dataframe(
        extreme_df[["date", "region", "precip_true", "pred_precip_xgb", "pred_precip_hybrid"]],
        use_container_width=True
    )

    fig = px.scatter(
        df, x="precip_true", y="pred_precip_hybrid", color="is_extreme_actual",
        title="Actual vs Predicted Precipitation (Hybrid Ensemble)",
        labels={"precip_true": "Actual (mm)", "pred_precip_hybrid": "Predicted (mm)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: ABOUT
# ---------------------------------------------------------------------------
def render_about():
    st.title("About this Dashboard")
    st.markdown(
        """
        This dashboard accompanies a dissertation on **UK Temperature & Precipitation Forecasting**.

        **Models evaluated:** Persistence, Random Forest, XGBoost, ARIMA, SARIMA,
        Holt-Winters, Transformer, Prophet (per-station), and a Hybrid Ensemble
        (classifier + specialist regressor for extreme rainfall detection).

        **Data source:** Daily weather station records across UK regions
        (England, Scotland, Wales, Northern Ireland).

        Built with Streamlit, Plotly, scikit-learn, and XGBoost.
        """
    )

# ---------------------------------------------------------------------------
# ROUTING LOGIC
# ---------------------------------------------------------------------------
if page == "🏠  Dashboard":
    render_dashboard()
elif page == "📊  Model Comparison":
    render_model_comparison()
elif page == "🌦  Seasonal Analysis":
    render_seasonal_analysis()
elif page == "📍  Regional Breakdown":
    render_regional_breakdown()
elif page == "⚠️  Extreme Events":
    render_extreme_events()
elif page == "ℹ️  About":
    render_about()
