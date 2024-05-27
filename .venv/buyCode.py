import requests
import yfinance as yf
import talipp.indicators as ta
from talipp.ohlcv import OHLCVFactory
import alpaca_trade_api as alp
import time as t
import schedule
from datetime import datetime, time
import pytz

# ******************** EDIT DURING TRIALS ***************************

buy_cap_base = 3000
nasdaq_smp_difference_constant = 0.00025

# *******************************************************************

condensed_stock_list = []
final_flagged_list = []

api = alp.REST('PKJIDRFO8XYMG1VJ2HZB',
               'tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3',
               'https://paper-api.alpaca.markets'
               )


# downloads all NYSE/NASDAQ tickers to list
def download_tickers():
    with open('condensedtickers.txt', 'r') as filehandle:
        for line in filehandle:
            ind_ticker = line[:-1]
            condensed_stock_list.append(ind_ticker)


# places a buy order
def place_order(ticker, notional):
    try:
        api.submit_order(symbol=ticker, notional=notional)
        print("bought " + ticker)
    except:
        data_close = yf.Ticker(ticker).history(period="5d", interval="5m")
        current_price = data_close['Close'].iloc[-1]

        quantity = int(notional/current_price)
        if quantity > 0:
            api.submit_order(symbol=ticker, qty=quantity)
            print("bought " + ticker)


# function to calculate RSI
def rsi_values(ticker):
    power = yf.Ticker(ticker)
    try:
        df = power.history(period="5d", interval="5m")
        df['RSI'] = ta.RSI(14, df['Close'])

        reformat_rsi = (df['RSI'].to_string()).split('\n')
        rsi_list = []
        for i in reformat_rsi:
            if i != "Datetime":
                rsi_value = (i.split(' '))[-1]
                if rsi_value == 'None':
                    return "invalid"
                else:
                    try:
                        float(rsi_value)
                    except ValueError:
                        return "invalid"
                    rsi_list.append(float(rsi_value))

        return [rsi_list[-3], rsi_list[-2], rsi_list[-1]]
    except IndexError:
        return "invalid"


# function to calculate TTM squeeze histogram value
def TTM_Squeeze_values(ticker):
    data_set = yf.Ticker(ticker)
    data_history = data_set.history(period='5d', interval='5m')
    ohlcv = OHLCVFactory.from_matrix2([
        data_history['Open'],
        data_history['High'],
        data_history['Low'],
        data_history['Close'],
        data_history['Volume']
    ])

    data_history['TTM'] = ta.TTM(20, input_values=ohlcv)
    ttm_values_list = []
    for ttm in data_history['TTM'].tail(5):
        if ttm is None:
            return "invalid"
        ttm_values_list.append(ttm.histogram)
    return ttm_values_list


# returns "buy", "sell", "invalid" or None based on rsi
def check_rsi(ticker):
    past_3_rsi = rsi_values(ticker)
    if past_3_rsi == "invalid":
        return "invalid"
    elif past_3_rsi[0] < 30 or past_3_rsi[1] < 30 or past_3_rsi[2] < 30:
        return "buy"
    elif past_3_rsi[0] > 70 or past_3_rsi[1] > 70 or past_3_rsi[2] > 70:
        return "sell"
    else:
        return None


# checks TTM squeeze value to return "buy" "sell" or "invalid"
def check_ttm(ticker):
    ttm_list = TTM_Squeeze_values(ticker)
    if ttm_list == "invalid":
        return "invalid"
    validity = True
    for item in ttm_list:
        if item is None:
            validity = False
    if validity:
        slope_a = ttm_list[1] - ttm_list[0]
        slope_b = ttm_list[2] - ttm_list[1]
        slope_c = ttm_list[3] - ttm_list[2]
        slope_d = ttm_list[4] - ttm_list[3]
        if slope_a < 0 and slope_b > 0 and slope_c > 0 and slope_d > 0:
            return "buy"
        elif slope_a > 0 and slope_b < 0 and slope_c < 0 and slope_d < 0:
            return "sell"
        else:
            return None
    else:
        return "invalid"


# returns a list of tickers that we own
def get_all_positions_by_symbol():

    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKJIDRFO8XYMG1VJ2HZB",
        "APCA-API-SECRET-KEY": "tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3"
    }

    response = requests.get(url, headers=headers)
    pos_list = response.text.split(',')
    final_ticker_list = []
    for data in pos_list:
        if "symbol" in data:
            symb_breakdown = data.split('"')
            final_ticker_list.append(symb_breakdown[-2])
    return final_ticker_list


def finish_time():
    current_finish_time = datetime.now()

    # Format the time in US format
    us_format_time = current_finish_time.strftime("%m/%d/%Y %H:%M:%S")

    # Print the time
    return us_format_time


def main_BUY_function():
    print("executing main_BUY...")
    for stock in condensed_stock_list:
        rsi_condition = check_rsi(stock)
        if rsi_condition == "invalid":
            continue
        ttm_condition = check_ttm(stock)
        if ttm_condition == "invalid":
            continue
        if rsi_condition == "buy" and ttm_condition == "buy":
            current_pos_list = get_all_positions_by_symbol()
            no_pre_own = True
            for pos in current_pos_list:
                if stock == pos:
                    no_pre_own = False
            if no_pre_own:
                print(stock + " flagged")
                global final_flagged_list
                final_flagged_list.append(stock)


def distribute_cash_and_execute(flagged_list):

    buy_cap = buy_cap_base*(nasdaq_smp())
    if buy_cap > 0:
        account = api.get_account()
        account_cash = float(account.cash)
        for stock in flagged_list:
            if account_cash >= buy_cap:
                try:
                    place_order(stock, buy_cap)
                    account_cash -= buy_cap
                except:
                    continue
    else:
        print(f"Nasdaq and S&P are down {nasdaq_smp()*100} % in past 20 mins, no buys")


def nasdaq_smp():  # changed dow to nasdaq

    # NASDAQ data
    nasdaq_data = yf.Ticker("^IXIC")
    nasdaq_data_history = nasdaq_data.history(period='5d', interval='1m')
    twenty_min_close = float(nasdaq_data_history['Close'].iloc[-20])
    one_min_close = float(nasdaq_data_history['Close'].iloc[-1])
    nasdaq_percent_diff = abs((one_min_close-twenty_min_close)/twenty_min_close)
    if twenty_min_close > one_min_close:
        nasdaq_percent_diff *= -1

    # SMP data
    sp_data = yf.Ticker("^GSPC")
    sp_data_history = sp_data.history(period='5d', interval='1m')
    sp_twenty_min_close = float(sp_data_history['Close'].iloc[-20])
    sp_one_min_close = float(sp_data_history['Close'].iloc[-1])
    sp_percent_diff = abs((sp_one_min_close - sp_twenty_min_close) / sp_twenty_min_close)
    if twenty_min_close > one_min_close:
        sp_percent_diff *= -1

    avg_diff = (nasdaq_percent_diff+sp_percent_diff)/2
    if avg_diff < nasdaq_smp_difference_constant:
        return avg_diff
    else:
        return 1 + avg_diff


first_run = True


def run_main_logic():

    start_time = t.time()
    global final_flagged_list
    final_flagged_list = []
    main_BUY_function()
    distribute_cash_and_execute(final_flagged_list)
    print(final_flagged_list)
    end_time = t.time()
    runtime = end_time - start_time
    print('runtime: ' + str(runtime) + ' seconds')
    print(finish_time())


# Schedule the main logic to run every 20 minutes from 9:30 AM to 4:00 PM
download_tickers()
schedule.every().day.at("10:10").do(run_main_logic)
schedule.every().day.at("15:11").do(schedule.clear)

target_time = time(10, 10)
target_timezone = pytz.timezone('US/Eastern')

first_iteration = True


while True:
    # Get the current time in EST
    current_time = datetime.now(target_timezone).time()

    # Get the current datetime in EST
    current_datetime = datetime.now(target_timezone).replace(microsecond=0, second=0)

    # Combine the current date with the target time
    target_datetime = current_datetime.replace(hour=target_time.hour, minute=target_time.minute)

    # Check if the current time is equal to the target time
    if current_datetime.time() == target_time:
        if first_iteration:
            print("Intiating 20 min schedule...")
            schedule.every(20).minutes.do(run_main_logic)
            first_iteration = False
# Keep the script running
    schedule.run_pending()
    t.sleep(1)
