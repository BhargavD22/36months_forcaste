# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
from google.cloud import bigquery
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"

# Set Streamlit page config for wide layout
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded",
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
def get_image_base64(path):
    """Encodes an image to a base64 string."""
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

encoded_string = get_image_base64(LOGO_PATH)

st.markdown(
    f"""
    <style>
        /* Enable smooth scrolling for bookmarks */
        html {{
            scroll-behavior: smooth;
        }}
        
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

        /* New card-like container style for KPIs */
        .kpi-card {{
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
            text-align: center;
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
        }}
        .kpi-label {{
            font-size: 1rem;
            color: #666;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #007bff;
            margin-bottom: 0;
        }}
        .kpi-delta {{
            font-size: 0.875rem;
            color: #28a745;
            margin-top: 0.5rem;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Title and Description ---
st.title("📈 Financial Forecasting Dashboard")
st.markdown("A **dynamic** application to analyze historical revenue data from **Google BigQuery** and forecast future trends using the **Prophet** model.")

with st.sidebar:
    # Add logo to the sidebar
    if encoded_string:
        st.image(LOGO_PATH, use_column_width=True)
    else:
        st.error(f"Logo file not found at {LOGO_PATH}")
        
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

    st.markdown("---")
    
    # --- Bookmarks Section ---
    st.header("🔖 Bookmarks")
    st.markdown(
        """
        [Core KPIs](#core-kpis)
        [Cumulative Revenue](#cumulative-revenue)
        [Historical Data](#historical-data)
        [Forecast Chart](#forecast-chart)
        [Forecast Table](#forecast-table)
        [Model Performance](#model-performance)
        [Time Series Components](#time-series-components)
        """,
        unsafe_allow_html=True
    )
    
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
        # Ensure df['ds'] is a datetime object before plotting.
        df['ds'] = pd.to_datetime(df['ds'])

        # Calculate 30-day moving average
        df['30_day_avg'] = df['y'].rolling(window=30).mean()

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
        
        # --- Calculate KPIs for comparison ---
        
        # Historical KPIs
        total_historical_revenue = df['y'].sum()
        avg_historical_revenue = df['y'].mean()
        highest_revenue_day_value = df['y'].max()
        highest_revenue_day_date = df.loc[df['y'].idxmax()]['ds'].strftime('%Y-%m-%d')
        lowest_revenue_day_value = df['y'].min()
        lowest_revenue_day_date = df.loc[df['y'].idxmin()]['ds'].strftime('%Y-%m-%d')
        
        # Forecasted KPIs
        forecast_df = forecast[forecast['ds'] > df['ds'].max()]
        total_forecasted_revenue = forecast_df['yhat_what_if'].sum()
        avg_forecasted_revenue = forecast_df['yhat_what_if'].mean()
        highest_forecasted_day_value = forecast_df['yhat_what_if'].max()
        highest_forecasted_day_date = forecast_df.loc[forecast_df['yhat_what_if'].idxmax()]['ds'].strftime('%Y-%m-%d')
        
        # Combined KPIs for overall view
        combined_df = pd.concat([df, forecast_df.rename(columns={'yhat_what_if': 'y'})[['ds', 'y']]])
        cagr = 0
        if len(combined_df) > 1:
            years = (combined_df['ds'].iloc[-1] - combined_df['ds'].iloc[0]).days / 365.25
            if years > 0:
                cagr = ((combined_df['y'].iloc[-1] / combined_df['y'].iloc[0]) ** (1/years) - 1) * 100

        # --- Display Core Revenue KPIs ---
        st.markdown('<div id="core-kpis"></div>', unsafe_allow_html=True)
        st.subheader("Core Revenue KPIs: Historical vs. Forecasted")
        
        col_hist, col_forecast = st.columns(2)
        
        with col_hist:
            st.markdown("### 📊 Historical")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Total Revenue</p>
                    <p class="kpi-value">${total_historical_revenue:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Average Daily Revenue</p>
                    <p class="kpi-value">${avg_historical_revenue:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Highest Day</p>
                    <p class="kpi-value">${highest_revenue_day_value:,.2f}</p>
                    <p class="kpi-delta" style="color: #666;">Date: {highest_revenue_day_date}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Lowest Day</p>
                    <p class="kpi-value">${lowest_revenue_day_value:,.2f}</p>
                    <p class="kpi-delta" style="color: #666;">Date: {lowest_revenue_day_date}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_forecast:
            st.markdown("### 🔮 Forecasted")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Total Projected Revenue ({forecast_months} mo)</p>
                    <p class="kpi-value">${total_forecasted_revenue:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Average Projected Revenue</p>
                    <p class="kpi-value">${avg_forecasted_revenue:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Highest Projected Day</p>
                    <p class="kpi-value">${highest_forecasted_day_value:,.2f}</p>
                    <p class="kpi-delta" style="color: #666;">Date: {highest_forecasted_day_date}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Projected CAGR</p>
                    <p class="kpi-value">{cagr:,.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # --- Cumulative Revenue Chart ---
        st.markdown('<div id="cumulative-revenue"></div>', unsafe_allow_html=True)
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

        # --- Historical Revenue Data with related KPIs ---
        st.markdown('<div id="historical-data"></div>', unsafe_allow_html=True)
        st.subheader("Historical Revenue Data")
        
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
              
        # --- Forecast Chart ---
        st.markdown('<div id="forecast-chart"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="forecast-table"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)
        st.subheader("📊 Model Performance")
        
        # Prepare data for comparison
        historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
        
        # Calculate new KPIs
        wape = np.sum(np.abs(historical_comparison['y'] - historical_comparison['yhat'])) / np.sum(np.abs(historical_comparison['y'])) * 100
        forecast_bias = np.mean(historical_comparison['yhat'] - historical_comparison['y'])

        # Create columns for side-by-side metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Mean Absolute Error (MAE)</p>
                    <p class="kpi-value">${np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat'])):,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Root Mean Squared Error (RMSE)</p>
                    <p class="kpi-value">${np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2)):,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">WAPE</p>
                    <p class="kpi-value">{wape:,.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">Forecast Bias</p>
                    <p class="kpi-value">${forecast_bias:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""
        **What are these metrics?**

        * **Mean Absolute Error (MAE)**: The average dollar amount the forecast was off by.
        * **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily, useful for spotting major misses.
        * **WAPE (Weighted Absolute Percentage Error)**: Provides a single percentage for overall accuracy, making it easy to interpret.
        * **Forecast Bias**: A positive value means the model is consistently over-forecasting, while a negative value indicates under-forecasting.
        """)
        
        st.markdown('<div id="time-series-components"></div>', unsafe_allow_html=True)
        st.subheader("📉 Time Series Components")
        st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
        components_fig = plot_components_plotly(model, forecast)
        st.plotly_chart(components_fig, use_container_width=True)

# --- Floating Chat (GLOBAL OVERLAY via Shadow DOM) ---
import base64, os
import streamlit.components.v1 as components

CHAT_ICON_PATH = "miralogo.png"

try:
    with open(CHAT_ICON_PATH, "rb") as _img:
        _CHAT_ICON_B64 = base64.b64encode(_img.read()).decode()
except Exception:
    _CHAT_ICON_B64 = ""  # falls back to emoji

overlay_html = """
<script>
(function () {
  // Create a Shadow DOM overlay in the PARENT page (so it’s never clipped/hidden)
  var canParent = false;
  try { canParent = !!window.parent && !!window.parent.document && !!window.parent.document.body; } catch (e) { canParent = false; }
  var hostDoc = canParent ? window.parent.document : document;

  var ROOT_ID = "mira-overlay-root";
  var hostDiv = hostDoc.getElementById(ROOT_ID);
  if (!hostDiv) {
    hostDiv = hostDoc.createElement("div");
    hostDiv.id = ROOT_ID;
    hostDoc.body.appendChild(hostDiv);
  }

  var shadow = hostDiv.shadowRoot || hostDiv.attachShadow({ mode: "open" });

  var ICON_B64 = "__ICON_B64__";
  var hasIcon = !!ICON_B64;

  var css = ''
  + '.mira-chat-toggle{position:fixed;bottom:18px;right:18px;width:56px;height:56px;border-radius:50%;'
  + 'background:#fff;border:2px solid #007bff;box-shadow:0 8px 24px rgba(0,0,0,.18);display:grid;place-items:center;cursor:pointer;'
  + 'z-index:9999999999;background-repeat:no-repeat;background-position:center;background-size:30px 30px;animation:mira-pulse 2s infinite;}'
  + (hasIcon ? ('.mira-chat-toggle{background-image:url(data:image/png;base64,' + ICON_B64 + ');} .mira-chat-toggle span{display:none;}')
              : ('.mira-chat-toggle span{font-size:22px;display:block;}'))
  + '@keyframes mira-pulse{0%{box-shadow:0 0 0 0 rgba(0,123,255,.45);}70%{box-shadow:0 0 0 16px rgba(0,123,255,0);}100%{box-shadow:0 0 0 0 rgba(0,123,255,0);}}'
  + '.mira-chat-modal{position:fixed;bottom:84px;right:18px;width:360px;max-width:96vw;height:480px;max-height:70vh;background:#fff;color:#111827;'
  + 'border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 24px 48px rgba(0,0,0,.18);display:none;flex-direction:column;overflow:hidden;z-index:9999999999;}'
  + '@media (max-width:480px){.mira-chat-modal{left:12px;right:12px;width:auto;height:70vh;}}'
  + '.mira-chat-header{background:#007bff;color:#fff;padding:10px 14px;font-weight:600;display:flex;align-items:center;justify-content:space-between;}'
  + '.mira-chat-header button{background:transparent;border:none;color:#fff;font-size:18px;cursor:pointer;}'
  + '.mira-chat-body{flex:1;background:#fafbfc;padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;}'
  + '.mira-msg{max-width:85%;padding:10px 12px;border-radius:12px;border:1px solid #e5e7eb;line-height:1.35;font-size:14px;white-space:pre-wrap;word-break:break-word;}'
  + '.mira-user{align-self:flex-end;background:#e9f3ff;border-color:#cfe5ff;} .mira-bot{align-self:flex-start;background:#f5f7fb;}'
  + '.mira-typing{align-self:flex-start;display:inline-flex;gap:6px;padding:8px 10px;border-radius:12px;border:1px solid #e5e7eb;background:#f5f7fb;color:#6b7280;font-size:14px;}'
  + '.mira-typing .dot{width:6px;height:6px;border-radius:50%;background:#6b7280;opacity:.6;animation:mira-blink 1.2s infinite;}'
  + '.mira-typing .dot:nth-child(2){animation-delay:.2s;} .mira-typing .dot:nth-child(3){animation-delay:.4s;}'
  + '@keyframes mira-blink{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}'
  + '.mira-chat-input{display:flex;gap:8px;padding:10px;border-top:1px solid #e5e7eb;background:#fff;}'
  + '.mira-chat-input input{flex:1;font-size:14px;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;outline:none;}'
  + '.mira-chat-input button{background:#007bff;color:#fff;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer;}'
  + '.mira-chat-input button:disabled{opacity:.65;cursor:not-allowed;}'
  + '.mira-table-wrap{overflow-x:auto;} .mira-table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px;}'
  + '.mira-table th,.mira-table td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;} .mira-table th{background:#f3f4f6;}';

  shadow.innerHTML = ''
    + '<style>' + css + '</style>'
    + '<button class="mira-chat-toggle" id="miraToggle" aria-label="Open chat"><span>💬</span></button>'
    + '<div class="mira-chat-modal" id="miraModal" role="dialog" aria-modal="true" aria-label="Mira Chat">'
    + '  <div class="mira-chat-header"><span>💬 Chat Assistant</span><button id="miraClose" aria-label="Close chat">✖</button></div>'
    + '  <div class="mira-chat-body" id="miraBody"></div>'
    + '  <div class="mira-chat-input"><input id="miraInput" type="text" placeholder="Ask your question..." /><button id="miraSend">Send</button></div>'
    + '</div>';

  // 👇 CHANGE #1: point to your Cloud Run proxy URL (no access key here)
  var API_URL = "https://mira-proxy-331663702828.us-central1.run.app/";

  var modal    = shadow.getElementById("miraModal");
  var toggle = shadow.getElementById("miraToggle");
  var closeB = shadow.getElementById("miraClose");
  var body   = shadow.getElementById("miraBody");
  var input  = shadow.getElementById("miraInput");
  var send   = shadow.getElementById("miraSend");

  function openChat(){ modal.style.display="flex"; setTimeout(function(){ input.focus(); }, 50); }
  function closeChat(){ modal.style.display="none"; }
  function scrollToBottom(){ body.scrollTop = body.scrollHeight; }

  function bubble(role, html){
    var div = document.createElement("div");
    div.className = "mira-msg " + (role === "user" ? "mira-user" : "mira-bot");
    if (typeof html === "string") div.innerText = html; else div.appendChild(html);
    body.appendChild(div); scrollToBottom();
  }

  function tableFromRows(rows){
    var table = document.createElement("table"); table.className="mira-table";
    var head = document.createElement("thead"); var htr = document.createElement("tr");
    var keys = Object.keys(rows[0] || {});
    keys.forEach(function(k){ var th=document.createElement("th"); th.textContent=k; htr.appendChild(th); });
    head.appendChild(htr); table.appendChild(head);
    var tb = document.createElement("tbody");
    rows.slice(0,10).forEach(function(r){
      var tr=document.createElement("tr");
      keys.forEach(function(k){
        var td=document.createElement("td"); var v=r[k];
        td.textContent = (typeof v === "number") ? v.toLocaleString() : (v == null ? "" : String(v));
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    var wrap=document.createElement("div"); wrap.className="mira-table-wrap"; wrap.appendChild(table); return wrap;
  }

  var busy=false, typingEl=null;
  function setBusy(state){
    busy=state; send.disabled=state; input.disabled=state;
    if(state){
      typingEl=document.createElement("div");
      typingEl.className="mira-typing";
      typingEl.innerHTML='<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
      body.appendChild(typingEl);
    } else if (typingEl){ typingEl.remove(); typingEl=null; }
    scrollToBottom();
  }

  async function sendMessage(){
    if (busy) return;
    var q = (input.value || "").trim(); if (!q) return;
    bubble("user", q); input.value=""; setBusy(true);

    try{
      // 👇 CHANGE #2: only Content-Type header; the proxy adds access-key
      var res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: q }),
        mode: "cors"
      });

      if (!res.ok){
        var txt = "";
        try{ txt = await res.text(); }catch(_){}
        bubble("bot", "API error " + res.status + (txt ? (": " + txt.slice(0,160)) : ""));
        return;
      }

      var data = {};
      try{ data = await res.json(); }catch(_){ data = {}; }

      var resp    = (data && data.response) ? data.response : {};
      var summary = resp.summerized || resp.summarized || "";
      var rows    = Array.isArray(resp.data) ? resp.data : [];

      if (summary) bubble("bot", summary);
      if (rows && rows.length) bubble("bot", tableFromRows(rows));
      if (!summary && (!rows || !rows.length)) bubble("bot", "No summary or data returned.");
    } catch (e){
      bubble("bot", "Request failed: " + (e && e.message ? e.message : ""));
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  toggle.addEventListener("click", function(){ (modal.style.display === "flex") ? closeChat() : openChat(); });
  closeB.addEventListener("click", closeChat);
  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", function(e){ if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); } });
})();
</script>
"""

# inject the icon data without needing f-string brace escaping
overlay_html = overlay_html.replace("__ICON_B64__", _CHAT_ICON_B64)

# Render with zero height; overlay is attached to parent DOM and floats globally
components.html(overlay_html, height=0, scrolling=False)
