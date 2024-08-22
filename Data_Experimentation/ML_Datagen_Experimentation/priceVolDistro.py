import yfinance as yf
import statsmodels.api as sm
import pandas as pd


def scale_mapping(a):
    return round((a + 6) / 18 * 8 + 1)


def range_mapping(b):
    if b < 0 or b > 8600:
        raise ValueError("Input must be between 0 and 8600")

    return min(int(b / 8.6) + 1, 1000)


# Read tickers from file
with open('condensedtickers.txt', 'r') as filehandle:
    ticker_list = [line.strip() for line in filehandle]

results = {}
high_beta_tickers = []
low_beta_tickers = []

# Assign beta values directly for specific tickers
assigned_betas = {
    'CHRD': 0.87,
    'CONL': 0,
    'CORZ': 1,
    'PMEC': 1,
    'VFS': 1.20,
    'ZJYL': 1,
    'DNTH': 1,
    'NLOP': 1,
    'ANRO': 1,
    'DEC': 0.22,
    'FGN': 1,
    'FIHL': 1,
    'KLG': 1,
    'MNR': 1,
    'SDHC': 1,
    'VSTS': 1,
    'AAPD': 0,
    'ABVX': 1.48,
    'AFIB': 0.01,
    'AIRE': 1,
    'ALCE': 0.25,
    'AMIX': 1,
    'API': 0.05,
    'BFRG': 1,
    'BGXX': 0.67,
    'CGON': 1,
    'COSM': 3.17,
    'CRCT': 0.03,
    'DTST': 0.75,
    'GDHG': 1,
    'GPCR': 1,
    'GRI': 1,
    'GUTS': 1,
    'HUBC': 1,
    'IVP': 1,
    'KXIN': 0.84,
    'KYTX': 1,
    'LICN': 1,
    'LUXH': 1,
    'LYRA': 0,
    'MNMD': 2.39,
    'MRVI': 0.01,
    'NB': 0.25,
    'NBBK': 1,
    'NVD': 0,
    'NVDD': 0,
    'NVDS': 0,
    'OABI': 1,
    'OMH': 1,
    'PRZO': 1,
    'RR': 1,
    'SARK': 0,
    'SFWL': 1,
    'SHV': 0.02,
    'SWIN': 1,
    'TCBP': 0.08,
    'TGL': 1,
    'THCH': 0.27,
    'TSLQ': 0,
    'TSLS': 0,
    'URNJ': 0,
    'USEA': 0.29,
    'WALD': 1,
    'WBUY': 1,
    'XBIL': 0,
}

percentage_list = []

for ticker in ticker_list:
    try:
        if ticker in assigned_betas:
            beta = assigned_betas[ticker]
        else:
            stock_data = yf.download(ticker, start='2019-07-01', end='2024-07-25', interval='1mo')
            market_data = yf.download('^GSPC', start='2019-07-01', end='2024-07-25', interval='1mo')

            # Check if data is empty
            if stock_data.empty or market_data.empty:
                print(f"No data found for {ticker}")
                continue

            stock_returns = stock_data['Close'].pct_change()
            market_returns = market_data['Close'].pct_change()

            # Concatenate and dropna
            returns = pd.concat([stock_returns, market_returns], axis=1).dropna()

            # Split back into stock and market returns
            stock_returns = returns.iloc[:, 0]
            market_returns = returns.iloc[:, 1]

            x = sm.add_constant(market_returns)
            model = sm.OLS(stock_returns, x).fit()

            beta = model.params[1]

        price = yf.Ticker(ticker).info.get("currentPrice", None)

        results[ticker] = {'beta': beta, 'price': price}

        volatility_rating = scale_mapping(beta)
        price_rating = range_mapping(price)
        percentage_list.append(ticker + ',' + str(volatility_rating) + ',' + str(price_rating))
        print(f"Ticker: {ticker}, Vol: {volatility_rating}, Price: {price_rating}")

        # Check for high and low beta values
        if beta > 5:
            high_beta_tickers.append(ticker)
        elif beta < 0:
            low_beta_tickers.append(ticker)

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

# Filter out tickers with no current price
filtered_results = {ticker: values for ticker, values in results.items() if values['price'] is not None}

# Find min and max values
min_price_ticker = min(filtered_results, key=lambda x: filtered_results[x]['price'])
max_price_ticker = max(filtered_results, key=lambda x: filtered_results[x]['price'])
min_beta_ticker = min(filtered_results, key=lambda x: filtered_results[x]['beta'])
max_beta_ticker = max(filtered_results, key=lambda x: filtered_results[x]['beta'])

print(f"Minimum Price: {filtered_results[min_price_ticker]['price']}, Ticker: {min_price_ticker}")
print(f"Maximum Price: {filtered_results[max_price_ticker]['price']}, Ticker: {max_price_ticker}")
print(f"Minimum Beta: {filtered_results[min_beta_ticker]['beta']}, Ticker: {min_beta_ticker}")
print(f"Maximum Beta: {filtered_results[max_beta_ticker]['beta']}, Ticker: {max_beta_ticker}")

print("High Beta Tickers (>5):", high_beta_tickers)
print("Low Beta Tickers (<0):", low_beta_tickers)

with open('pricevoldistrolist.txt', 'w') as file:  # updates with new stock
    for obj in percentage_list:
        file.write(obj + "\n")

# Minimum Price: 0.0003, Ticker: BTTX
# Maximum Price: 8597.355, Ticker: NVR
# Minimum Beta: -5.8530447239364385, Ticker: IVT
# Maximum Beta: 11.369540988972528, Ticker: PRTG
