import alpaca_trade_api as alp
import yfinance as yf
import pytz
import schedule
import time as t
from datetime import datetime, time

# ******************** EDIT DURING TRIALS ***************************

buy_cap_base = 3000
nasdaq_smp_difference_constant = 0
buy_spacer = 1.00045

# *******************************************************************


api = alp.REST('PKJIDRFO8XYMG1VJ2HZB',
               'tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3',
               'https://paper-api.alpaca.markets'
               )


def get_current_price(ticker):
    data_close = yf.Ticker(ticker).history(period="5d", interval="1m")
    try:
        current_price = data_close['Close'].iloc[-1]
        return current_price
    except:
        return -1


def main_watch_function():
    print("watching")
    watchlist = []
    timer_list = []
    price_list = []

    while len(watchlist) != len(timer_list) or len(watchlist) != len(price_list) or len(timer_list) != len(price_list):
        with open('watchlist.txt', 'r') as filehandle:  # grabs watchlist
            for line in filehandle:
                watch_ticker = line[:-1]
                watchlist.append(watch_ticker)

        with open('timers.txt', 'r') as filehandle:  # grabs timers
            for line in filehandle:
                time = line[:-1]
                timer_list.append(time)

        with open('prices.txt', 'r') as filehandle:  # grabs current price list
            for line in filehandle:
                price = line[:-1]
                price_list.append(price)

    with open('watchlist.txt', 'r') as filehandle:  # grabs watchlist
        for line in filehandle:
            watch_ticker = line[:-1]
            watchlist.append(watch_ticker)

    with open('timers.txt', 'r') as filehandle:  # grabs timers
        for line in filehandle:
            time = line[:-1]
            timer_list.append(time)

    with open('prices.txt', 'r') as filehandle:  # grabs current price list
        for line in filehandle:
            price = line[:-1]
            price_list.append(price)

    if len(watchlist) > 0:
        final_index = len(watchlist) - 1
        universal_index = 0
        for stock in watchlist:
            if time_in_minutes() - int(timer_list[universal_index]) > 20:
                watchlist.pop(universal_index)
                timer_list.pop(universal_index)
                price_list.pop(universal_index)
                rewrite_all_list(watchlist, timer_list, price_list)

            try:
                if float(get_current_price(stock)) >= (buy_spacer * (float(price_list[universal_index]))):
                    place_order(stock)
                    watchlist.pop(universal_index)
                    timer_list.pop(universal_index)
                    price_list.pop(universal_index)
                    rewrite_all_list(watchlist, timer_list, price_list)
            except:
                print("Index error")
                print(universal_index)
                print(watchlist)
                print(price_list)

            if universal_index == final_index:
                break
            else:
                universal_index += 1


def time_in_minutes():
    target_timezone_1 = pytz.timezone('US/Eastern')
    current_time_1 = datetime.now(target_timezone_1).time()
    split_time = str(current_time_1).split(':')
    return (int(split_time[0])*60)+int(split_time[1])


# places a buy order
def place_order(ticker):
    buy_cap = buy_cap_base * (nasdaq_smp())
    if buy_cap > 0:
        account = api.get_account()
        account_cash = float(account.cash)
        if account_cash >= buy_cap:
            try:
                try:
                    api.submit_order(symbol=ticker, notional=buy_cap)
                    print("bought " + ticker)
                    add_to_queue(ticker)
                    target_timezone_1 = pytz.timezone('US/Eastern')
                    current_time_1 = datetime.now(target_timezone_1).time()
                    print(current_time_1)
                except:
                    data_close = yf.Ticker(ticker).history(period="5d", interval="5m")
                    current_price = data_close['Close'].iloc[-1]

                    quantity = int(buy_cap / current_price)
                    if quantity > 0:
                        api.submit_order(symbol=ticker, qty=quantity)
                        print("bought " + ticker)
                        add_to_queue(ticker)
                        target_timezone_1 = pytz.timezone('US/Eastern')
                        current_time_1 = datetime.now(target_timezone_1).time()
                        print(current_time_1)
            except:
                print(ticker + " fail to buy")
                target_timezone_1 = pytz.timezone('US/Eastern')
                current_time_1 = datetime.now(target_timezone_1).time()
                print(current_time_1)
    else:
        print(f"Nasdaq and S&P are down {nasdaq_smp()} % in past 20 mins, no buys")


def nasdaq_smp():  # changed dow to nasdaq

    # NASDAQ data
    nasdaq_data = yf.Ticker("^IXIC")
    nasdaq_data_history = nasdaq_data.history(period='5d', interval='1m')
    twenty_min_close = float(nasdaq_data_history['Close'].iloc[-20])
    one_min_close = float(nasdaq_data_history['Close'].iloc[-1])
    nasdaq_percent_diff = abs((one_min_close - twenty_min_close) / twenty_min_close)
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

    avg_diff = (nasdaq_percent_diff + sp_percent_diff) / 2
    if avg_diff < nasdaq_smp_difference_constant:
        return avg_diff
    else:
        return 1 + avg_diff


def rewrite_all_list(watchlist, timer_list, price_list):

    with open('watchlist.txt', 'w') as file:  # updates with new stock
        for ticker in watchlist:
            file.write(ticker + "\n")

    with open('timers.txt', 'w') as file:  # updates with new timer
        for time in timer_list:
            file.write(str(time) + "\n")

    with open('prices.txt', 'w') as file:  # updates with new price
        for price in price_list:
            file.write(price + "\n")


def add_to_queue(ticker):  # watchlist code
    queue_list = []
    new_in_queue = ticker + " BUY"

    with open('recordsqueue.txt', 'r') as filehandle:  # grabs current watchlist
        for line in filehandle:
            in_queue = line[:-1]
            queue_list.append(in_queue)

    queue_list.append(new_in_queue)

    with open('recordsqueue.txt', 'w') as file:  # updates with new stock
        for obj in queue_list:
            file.write(obj + "\n")


schedule.every().day.at("10:10").do(main_watch_function)
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
            print("Intiating 1 min schedule...")
            schedule.every(1).minutes.do(main_watch_function)
            first_iteration = False
# Keep the script running
    schedule.run_pending()
    t.sleep(1)