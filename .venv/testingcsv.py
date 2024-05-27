import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates

# Create a new directory called png_files1
os.makedirs('png_files1', exist_ok=True)

# Create a file uploader widget
data = pd.read_csv('2024-05-14.csv')
for index, row in data.iterrows():
    ticker, flag_time, buy_time = row
    # Get the price vs time data from Yahoo Finance
    start_time = datetime(2024, 5, 14, 9, 30) + timedelta(minutes=int(flag_time) - 30)
    end_time = datetime(2024, 5, 14, 9, 30) + timedelta(minutes=int(buy_time) + 30)
    stock_data = yf.download(ticker, start=start_time, end=end_time, interval='1m')
    # Convert flag_time and buy_time to datetime format
    base_time = datetime(2024, 5, 14, 9, 30)
    flag_time = pd.to_datetime(base_time + timedelta(minutes=int(flag_time)))
    buy_time = pd.to_datetime(base_time + timedelta(minutes=int(buy_time)))
    # Create the graph
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'])  # Plot the price vs time data

    # Set the x-axis limits to the minimum and maximum dates in the data
    ax.set_xlim([stock_data.index.min(), stock_data.index.max()])
    # Draw the vertical lines
    print(flag_time, ticker)
    ax.axvline(x=flag_time, color='red', linestyle='--')
    ax.axvline(x=buy_time, color='red', linestyle='--')
    # Save the graph as a PNG file
    png_file = f'png_files1/{ticker}.jpeg'
    fig.savefig(png_file)