# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
from google.cloud import bigquery
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"

# Set Streamlit page config for wide layout and dark theme
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded"
)

# === Function to connect and fetch data from BigQuery === #
@st.cache_data
def get_bigquery_data():
    # Set up the BigQuery client using Streamlit secrets
    client = bigquery.Client.from_service_account_info(st.secrets["connections"]["gcp_service_account"])

    # Your SQL query for BigQuery
    # Note: Assumes the table has 'ds' and 'y' columns.
    query = """
        SELECT
            *
        FROM
            `mss-data-engineer-sandbox.financial_streamlit.financial_forecasting`
    """
    
    # Execute the query and load results into a DataFrame
    query_job = client.query(query)
    df = query_job.result().to_dataframe()
    return df

# === Streamlit App UI === #

# --- Custom CSS for Styling ---
# Read the logo image and encode it to Base64
with open(LOGO_PATH, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Apply custom theme and background */
        .stApp {{
            background-color: #1a1a2e;
            color: #d1d1d1;
        }}

        /* Style for the main container */
        .st-emotion-cache-1r4qj8m {{
            background-color: #2e2e4e;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }}

        /* Style for the logo container */
        .logo-container {{
            text-align: right;
        }}
        .logo-container img {{
            height: 60px;
            width: auto;
            border-radius: 8px;
        }}

        /* Style for headers */
        h1, h2, h3, h4, h5, h6 {{
            color: #ff9900; /* A contrasting color */
        }}

        /* Style for metrics */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: #ff9900;
        }}

        /* Style for the slider */
        .stSlider .st-emotion-cache-6q9m8y e16fv1ov3 {{
            background-color: #ff9900;
        }}
        
        /* Style the tabs */
        .stTabs [role="tablist"] button {{
            background-color: #2e2e4e;
            color: #d1d1d1;
            border-bottom: 3px solid transparent;
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            color: #ff9900;
            border-bottom: 3px solid #ff9900;
        }}
        
        /* Style for the dataframe */
        .dataframe {{
            border-radius: 8px;
        }}
        
        /* Style for the download button */
        .stDownloadButton button {{
            background-color: #ff9900;
            color: #1a1a2e;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover {{
            background-color: #e68a00;
            color: #1a1a2e;
        }}

    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Title and Description ---
# Use columns to place the title and logo side-by-side
title_col, logo_col = st.columns([3, 1])

with title_col:
    st.title("📈 Financial Forecasting Dashboard")
    st.markdown("A **dynamic** application to analyze historical revenue data from **Google BigQuery** and forecast future trends using the **Prophet** model.")

with logo_col:
    st.markdown(
        f'<div class="logo-container"><img src="data:image/png;base64,{encoded_string}" alt="Miracle Software Systems Logo"></div>',
        unsafe_allow_html=True
    )

# --- Interactive Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    forecast_months = st.slider("Select number of months to forecast:", min_value=1, max_value=60, value=36)
    forecast_period_days = forecast_months * 30  # Prophet uses days

# --- Main Content Area ---
st.header("Data & Analysis")

# Load data
with st.spinner("Connecting to Google BigQuery and fetching data..."):
    df = get_bigquery_data()

# Create main content tabs
tab1, tab2 = st.tabs(["📊 Forecast", "📈 Model Performance"])

with tab1:
    # --- Historical Data Plot ---
    st.subheader("Historical Revenue Data")
    # Ensure df['ds'] is a datetime object before plotting.
    df['ds'] = pd.to_datetime(df['ds'])
    st.line_chart(df.set_index('ds')['y'])

    # Fit Prophet model
    model = Prophet()
    model.fit(df)

    # Make forecast
    future = model.make_future_dataframe(periods=forecast_period_days)
    forecast = model.predict(future)
    
    # --- FIX: Convert 'ds' column to datetime to avoid TypeError ---
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    
    # --- Forecast Chart ---
    st.subheader(f"🔮 Forecasted Revenue ({forecast_months} Months)")

    # Separate historical and forecast parts
    historical = forecast[forecast['ds'] <= df['ds'].max()]
    future_forecast = forecast[forecast['ds'] > df['ds'].max()]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=historical['ds'], y=historical['yhat'],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'], y=future_forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=3, dash='dash')
    ))

    # Confidence interval shading
    fig.add_trace(go.Scatter(
        x=list(future_forecast['ds']) + list(future_forecast['ds'])[::-1],
        y=list(future_forecast['yhat_upper']) + list(future_forecast['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=f"Forecasted Revenue for Next {forecast_months} Months",
        xaxis_title="Date",
        yaxis_title="Revenue",
        template="plotly_dark", # Use a dark theme for Plotly
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast Table and Download ---
    st.subheader(f"🧾 {forecast_months}-Month Forecast Table")
    st.dataframe(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).rename(
            columns={
                "ds": "Date",
                "yhat": "Predicted Revenue",
                "yhat_lower": "Lower Bound",
                "yhat_upper": "Upper Bound"
            }
        )
    )

    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).to_csv(index=False)
    st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")


with tab2:
    st.subheader("📊 Forecast Accuracy")
    
    # Prepare data for comparison
    historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
    
    # Get the average of historical data to calculate percentage error
    average_y = df['y'].mean()

    # Create two columns for side-by-side metrics
    col1, col2 = st.columns(2)

    with col1:
        # Calculate Mean Absolute Error (MAE) and percentage
        mae = np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat']))
        mae_percent = (mae / average_y) * 100
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}", f"{mae_percent:,.2f}% of Average Revenue")

    with col2:
        # Calculate Root Mean Squared Error (RMSE) and percentage
        rmse = np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2))
        rmse_percent = (rmse / average_y) * 100
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}", f"{rmse_percent:,.2f}% of Average Revenue")
    
    st.markdown("""
    **What are these metrics?**

    * **Mean Absolute Error (MAE)**: This represents the average dollar amount your forecast was off by for each data point. The percentage value shows this error relative to your average historical revenue.

    * **Root Mean Squared Error (RMSE)**: This metric also measures the average error in dollars, but it penalizes larger errors more heavily. It's useful for understanding if your model had a few very large forecasting misses. The percentage value shows this error relative to your average historical revenue.
    """)

    st.subheader("📉 Time Series Components")
    st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
    components_fig = plot_components_plotly(model, forecast)
    st.plotly_chart(components_fig, use_container_width=True)
