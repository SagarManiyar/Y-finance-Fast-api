import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime
import pandas as pd
import os

# Set page title
st.title(" Stock Candlestick Charts for 5 Companies")

# List of companies
companies = ["RTX", "TSM", "YUM", "FXI", "PG"]


# Set default date range
today = datetime.date.today()
start_date = today - datetime.timedelta(days=180)  # last 6 months

# File to store stock data
csv_file = "stocks_data.csv"

# Fetch and store stock data if CSV not already created
csv_is_stale = False
if os.path.exists(csv_file):
    import time
    file_age_hours = (time.time() - os.path.getmtime(csv_file)) / 3600
    csv_is_stale = file_age_hours > 24

if not os.path.exists(csv_file) or csv_is_stale:
    all_data = []
    for ticker in companies:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=today, interval="1d")
        if not df.empty:
            df = df.reset_index()
            df["Company"] = ticker
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data)
        final_df.to_csv(csv_file, index=False)

# Read back from CSV
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Sidebar: company selector
    selected_company = st.sidebar.selectbox("Select a company:", companies)

    company_data = df[df["Company"] == selected_company]

    if company_data.empty:
        st.warning(f"No stock data available for {selected_company}.")
    else:
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=company_data["Date"],
            open=company_data['Open'],
            high=company_data['High'],
            low=company_data['Low'],
            close=company_data['Close'],
            name="Candlestick"
        )])

        fig_candle.update_layout(
            title=f"{selected_company} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )

        st.plotly_chart(fig_candle, use_container_width=True)

else:
    st.error("No stock data available.")