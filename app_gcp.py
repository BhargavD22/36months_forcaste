# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
from google.cloud import bigquery
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"

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

# --- Custom CSS for Logo ---
st.markdown(
    f"""
    <style>
        .logo-container {{
            position: fixed;
            top: 10px; /* Distance from the top */
            right: 10px; /* Distance from the right */
            z-index: 1000; /* Ensure it's above other content */
        }}
        .logo-container img {{
            height: 60px; /* Adjust logo height as needed */
            width: auto;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Add Logo HTML ---
st.markdown(
    f'<div class="logo-container"><img src="{LOGO_PATH}" alt="Miracle Software Systems Logo"></div>',
    unsafe_allow_html=True
)

st.title("📈 Financial Forecasting Application")
st.markdown("This app retrieves financial data from Google BigQuery ✔ and forecasts future revenue.")

# User selects forecast months
forecast_months = st.slider("Select number of months to forecast:", min_value=1, max_value=60, value=36)
forecast_period_days = forecast_months * 30  # Prophet uses days

# Load data
with st.spinner("Connecting to Google BigQuery and fetching data..."):
    df = get_bigquery_data()

# Check and preview data
st.subheader("📊 Historical Data")
df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format
st.line_chart(df.set_index('ds')['y'])

# Fit Prophet model
model = Prophet()
model.fit(df)

# Make forecast
future = model.make_future_dataframe(periods=forecast_period_days)
forecast = model.predict(future)

# === Create Tabs === #
tab1, tab2 = st.tabs(["📊 Forecast", "📈 Model Performance"])

with tab1:
    # === Best fit forecast chart with shaded confidence interval === #
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
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Forecast Table === #
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

    # === Export as CSV === #
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).to_csv(index=False)
    st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")


with tab2:
    st.subheader("📊 Forecast Accuracy")
    # Calculate Mean Absolute Error (MAE)
    historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
    mae = np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat']))
    st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    
    st.markdown("""
    **What is MAE?**
    It's the average absolute difference between the actual historical values and the model's predictions. A lower number indicates a more accurate model.
    """)

    st.subheader("📉 Time Series Components")
    st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
    components_fig = plot_components_plotly(model, forecast)
    st.plotly_chart(components_fig, use_container_width=True)
