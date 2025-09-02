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

# Set Streamlit page config for wide layout and light theme
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded",
    # The 'theme' argument has been removed to fix the TypeError.
    # The custom CSS below handles the app's visual theme.
)

# === Function to connect and fetch data from BigQuery === #
@st.cache_data
def get_bigquery_data():
    """
    Connects to Google BigQuery, fetches financial forecasting data,
    and returns it as a pandas DataFrame.
    """
    try:
        # Set up the BigQuery client using Streamlit secrets
        client = bigquery.Client.from_service_account_info(st.secrets["connections"]["gcp_service_account"])

        # Your SQL query for BigQuery
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
    except Exception as e:
        st.error(f"Error fetching data from BigQuery: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

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
    
    st.subheader("Forecast Period")
    forecast_months = st.slider("Select number of months to forecast:", min_value=1, max_value=60, value=36)
    forecast_period_days = forecast_months * 30  # Prophet uses days

    st.subheader("Model Configuration")
    confidence_interval = st.slider("Confidence Interval (%)", min_value=80, max_value=99, value=90, step=1) / 100
    
    st.markdown("**Seasonality Controls**")
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)

    st.subheader("What-if Scenario Analysis")
    what_if_change = st.number_input("Future Revenue Change (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.5, help="Enter a percentage change to simulate a what-if scenario. Ex: 10 for a 10% increase.")
    
# --- Main Content Area ---
st.header("Data & Analysis")

# Load data
with st.spinner("Connecting to Google BigQuery and fetching data..."):
    df = get_bigquery_data()

# Check if data was loaded successfully
if df.empty:
    st.warning("No data available to display. Please check your data source and credentials.")
else:
    # Create main content tabs
    tab1, tab2 = st.tabs(["📊 Forecast", "📈 Model Performance"])

    with tab1:
        # --- Historical Data Plot ---
        st.subheader("Historical Revenue Data")
        # Ensure df['ds'] is a datetime object before plotting.
        df['ds'] = pd.to_datetime(df['ds'])

        # Calculate 30-day moving average
        df['30_day_avg'] = df['y'].rolling(window=30).mean()

        fig_historical = go.Figure()
        fig_historical.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color='blue', width=2)
        ))
        fig_historical.add_trace(go.Scatter(
            x=df['ds'], y=df['30_day_avg'],
            mode='lines',
            name='30-Day Moving Average',
            line=dict(color='green', width=3)
        ))
        fig_historical.update_layout(
            title="Daily Revenue with 30-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Revenue",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_historical, use_container_width=True)

        # Fit Prophet model with user-defined seasonality
        model = Prophet(weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality)
        model.fit(df)

        # Make forecast with user-defined confidence interval
        future = model.make_future_dataframe(periods=forecast_period_days)
        forecast = model.predict(future)
        
        # --- Convert 'ds' column to datetime to avoid TypeError ---
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # --- Apply what-if scenario to the forecast ---
        forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100)
        
        # --- Calculate new KPIs ---
        total_historical_revenue = df['y'].sum()
        total_forecasted_revenue = forecast[forecast['ds'] > df['ds'].max()]['yhat_what_if'].sum()
        
        avg_historical_revenue = df['y'].mean()
        avg_forecasted_revenue = total_forecasted_revenue / forecast_period_days
        
        highest_revenue_day_date = df.loc[df['y'].idxmax()]['ds'].strftime('%Y-%m-%d')
        highest_revenue_day_value = df['y'].max()
        
        lowest_revenue_day_date = df.loc[df['y'].idxmin()]['ds'].strftime('%Y-%m-%d')
        lowest_revenue_day_value = df['y'].min()

        # Calculate MoM and YoY growth
        df_monthly = df.set_index('ds').resample('M').sum()
        mom_growth = 0
        yoy_growth = 0
        if len(df_monthly) >= 2:
            mom_growth = ((df_monthly['y'].iloc[-1] - df_monthly['y'].iloc[-2]) / df_monthly['y'].iloc[-2]) * 100
        if len(df_monthly) >= 12:
            last_month_year = df_monthly.index[-1].year
            last_month_month = df_monthly.index[-1].month
            last_year_value = df_monthly[(df_monthly.index.year == last_month_year - 1) & (df_monthly.index.month == last_month_month)]
            if not last_year_value.empty:
                yoy_growth = ((df_monthly['y'].iloc[-1] - last_year_value['y'].iloc[0]) / last_year_value['y'].iloc[0]) * 100

        # Calculate CAGR
        if len(df) > 1:
            years = (df['ds'].iloc[-1] - df['ds'].iloc[0]).days / 365.25
            if years > 0:
                cagr = ((df['y'].iloc[-1] / df['y'].iloc[0]) ** (1/years) - 1) * 100
            else:
                cagr = 0
        else:
            cagr = 0

        # --- Display KPIs in a multi-column layout ---
        st.subheader("Core Revenue KPIs")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="**Total Historical Revenue**", value=f"${total_historical_revenue:,.2f}")
        with col2:
            st.metric(label=f"**Total Forecasted Revenue ({forecast_months} mo)**", value=f"${total_forecasted_revenue:,.2f}")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric(label="**Average Revenue (Historical)**", value=f"${avg_historical_revenue:,.2f}")
        with col4:
            st.metric(label="**Highest Revenue Day**", value=f"${highest_revenue_day_value:,.2f}", delta=f"Date: {highest_revenue_day_date}")
        with col5:
            st.metric(label="**Lowest Revenue Day**", value=f"${lowest_revenue_day_value:,.2f}", delta=f"Date: {lowest_revenue_day_date}")
        
        st.markdown("---")
        st.subheader("Growth & Trend KPIs")
        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric(label="**Month-over-Month Growth**", value=f"{mom_growth:,.2f}%", delta="N/A" if mom_growth == 0 else (f"{mom_growth:,.2f}%"))
        with col7:
            st.metric(label="**Year-over-Year Growth**", value=f"{yoy_growth:,.2f}%", delta="N/A" if yoy_growth == 0 else (f"{yoy_growth:,.2f}%"))
        with col8:
            st.metric(label="**Historical CAGR**", value=f"{cagr:,.2f}%")

        # --- Cumulative Revenue Chart ---
        st.subheader("📈 Cumulative Revenue Trend")
        df['cumulative_revenue'] = df['y'].cumsum()
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=df['ds'], y=df['cumulative_revenue'],
            mode='lines',
            name='Cumulative Revenue',
            line=dict(color='purple', width=3)
        ))
        fig_cumulative.update_layout(
            title="Cumulative Revenue Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)
            
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
            x=future_forecast['ds'], y=future_forecast['yhat_what_if'],
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
            name=f"{confidence_interval*100:.0f}% Confidence Interval"
        ))

        fig.update_layout(
            title=f"Forecasted Revenue for Next {forecast_months} Months",
            xaxis_title="Date",
            yaxis_title="Revenue",
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Forecast Table and Download ---
        st.subheader(f"🧾 {forecast_months}-Month Forecast Table")
        st.dataframe(
            forecast[['ds', 'yhat_what_if', 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).rename(
                columns={
                    "ds": "Date",
                    "yhat_what_if": "Predicted Revenue",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound"
                }
            )
        )

        csv = forecast[['ds', 'yhat_what_if', 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).to_csv(index=False)
        st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")


    with tab2:
        st.subheader("📊 Model Performance")
        
        # Prepare data for comparison
        historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
        
        # Get the average of historical data to calculate percentage error
        average_y = df['y'].mean()

        # Calculate new KPIs
        wape = np.sum(np.abs(historical_comparison['y'] - historical_comparison['yhat'])) / np.sum(np.abs(historical_comparison['y'])) * 100
        forecast_bias = np.mean(historical_comparison['yhat'] - historical_comparison['y'])

        # Create columns for side-by-side metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("**Mean Absolute Error (MAE)**", f"${np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat'])):,.2f}")
        with col2:
            st.metric("**Root Mean Squared Error (RMSE)**", f"${np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2)):,.2f}")
        with col3:
            st.metric("**WAPE**", f"{wape:,.2f}%")
        with col4:
            st.metric("**Forecast Bias**", f"${forecast_bias:,.2f}")

        st.markdown("""
        **What are these metrics?**

        * **Mean Absolute Error (MAE)**: The average dollar amount the forecast was off by.
        * **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily, useful for spotting major misses.
        * **WAPE (Weighted Absolute Percentage Error)**: Provides a single percentage for overall accuracy, making it easy to interpret.
        * **Forecast Bias**: A positive value means the model is consistently over-forecasting, while a negative value indicates under-forecasting.
        """)

        st.subheader("📉 Time Series Components")
        st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
        components_fig = plot_components_plotly(model, forecast)
        st.plotly_chart(components_fig, use_container_width=True)
