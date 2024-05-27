import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from io import BytesIO

# Set the start and end dates and times
start_date = dt.datetime(2024, 5, 22, 9, 30)
end_date = dt.datetime(2024, 5, 22, 16)

# Get the data from Yahoo Finance
aapl_data = yf.download('AAPL', start=start_date, end=end_date, interval='1m')
tsla_data = yf.download('TSLA', start=start_date, end=end_date, interval='1m')

# Create a function to generate the graph
def generate_graph(data, ticker):
    fig, ax = plt.subplots()
    ax.plot(data['Close'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price ($)')
    ax.set_title(f'{ticker} Stock Price on May 22, 2024')
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()

# Create the Streamlit app
st.title("Stock Price Graphs")

# Add a dropdown menu to select the ticker
ticker_selection = st.selectbox('Select Ticker', ['AAPL', 'TSLA'])

# Display the selected graph
if ticker_selection == 'AAPL':
    st.image(generate_graph(aapl_data, 'AAPL'))
else:
    st.image(generate_graph(tsla_data, 'TSLA'))