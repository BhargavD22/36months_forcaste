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
LOGO_PATH = "miracle-logo-light.png" # Changed to the light logo

# Set Streamlit page config for wide layout and light theme
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded",
    theme="light" # Hardcode to light theme
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
        
        /* Apply custom theme and background for light mode */
        .stApp {{
            background-color: #f0f2f6;
            color: #333333;
        }}

        /* Style for the main container */
        .st-emotion-cache-1r4qj8m {{
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }}

        /* Style for headers */
        h1, h2, h3, h4, h5, h6 {{
            color: #007bff; /* A contrasting color for light mode */
        }}

        /* Style for metrics */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: #007bff;
        }}

        /* Style for the slider */
        .stSlider .st-emotion-cache-6q9m8y e16fv1ov3 {{
            background-color: #007bff;
        }}
        
        /* Style the tabs */
        .stTabs [role="tablist"] button {{
            background-color: #ffffff;
            color: #333333;
            border-bottom: 3px solid transparent;
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            color: #007bff;
            border-bottom: 3px solid #007bff;
        }}
        
        /* Style for the dataframe */
        .dataframe {{
            border-radius: 8px;
        }}
        
        /* Style for the download button */
        .stDownloadButton button {{
            background-color: #007bff;
            color: #ffffff;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover {{
            background-color: #0056b3;
            color: #ffffff;
        }}

    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Title and Description ---
st.title("📈 Financial Forecasting Dashboard")
st.markdown("A **dynamic** application to analyze historical revenue data from **Google BigQuery** and forecast future trends using the **Prophet** model.")

# --- Interactive Sidebar for Controls ---
with st.sidebar:
    # Add logo to the sidebar
    st.image(LOGO_PATH, use_column_width=True)
    st.header("⚙️ Settings")
    
    # Theme toggle is removed
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
    
    # --- Convert 'ds' column to datetime to avoid TypeError ---
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
        template="plotly_white", # Use the white theme for Plotly
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
        st.metric("**Mean Absolute Error (MAE)**", f"${np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat'])):,.2f}", f"{(np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat'])) / average_y) * 100:,.2f}% of Average Revenue")

    with col2:
        # Calculate Root Mean Squared Error (RMSE) and percentage
        st.metric("**Root Mean Squared Error (RMSE)**", f"${np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2)):,.2f}", f"{(np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2)) / average_y) * 100:,.2f}% of Average Revenue")
    
    st.markdown("""
    **What are these metrics?**

    * **Mean Absolute Error (MAE)**: This represents the average dollar amount your forecast was off by for each data point. The percentage value shows this error relative to your average historical revenue.

    * **Root Mean Squared Error (RMSE)**: This metric also measures the average error in dollars, but it penalizes larger errors more heavily. It's useful for understanding if your model had a few very large forecasting misses. The percentage value shows this error relative to your average historical revenue.
    """)

    st.subheader("📉 Time Series Components")
    st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
    components_fig = plot_components_plotly(model, forecast)
    st.plotly_chart(components_fig, use_container_width=True)
