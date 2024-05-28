import requests
import schedule
import time as t
import alpaca_trade_api as alp
import yfinance as yf
import pytz
from datetime import datetime, time

# ******************** EDIT DURING TRIALS ***************************

normal_stop_loss_constant = -0.0025  # updated 3/3 10:10 PM
trailing_stop_loss_constant = 0.003  # always positive value

# *******************************************************************
our_stocks_list = []
our_prices_list = []

api = alp.REST('PKJIDRFO8XYMG1VJ2HZB',
               'tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3',
               'https://paper-api.alpaca.markets'
               )


def main_SELL_function():
    start_time = t.time()

    print("executing main_SELL...")
    current_pos_list = get_all_positions_by_symbol()

    for position in current_pos_list:
        pre_own = False
        for stock in our_stocks_list:
            if stock == position:
                pre_own = True
        if pre_own:
            continue
        else:
            our_stocks_list.append(position)

    for position in current_pos_list:
        url = "https://paper-api.alpaca.markets/v2/positions/" + position

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": "PKJIDRFO8XYMG1VJ2HZB",
            "APCA-API-SECRET-KEY": "tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3"
        }

        response = requests.get(url, headers=headers)
        pos_list = response.text.split(',')
        final_pl = 0
        for data in pos_list:
            if "unrealized_plpc" in data:
                pl_breakdown = data.split('"')
                final_pl = float(pl_breakdown[-2])

        # buffer statement
        if final_pl < 0:
            if final_pl <= normal_stop_loss_constant:
                close_position(position)
        else:  # sell with trailing stop
            sell_decision = trailing_stop_loss(position)
            if sell_decision:
                close_position(position)

    end_time = t.time()
    print(f"runtime: {end_time-start_time} seconds")


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


def close_position(ticker):
    url = "https://paper-api.alpaca.markets/v2/positions/" + ticker + "?percentage=100"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKJIDRFO8XYMG1VJ2HZB",
        "APCA-API-SECRET-KEY": "tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3"
    }

    requests.delete(url, headers=headers)
    print("sold " + ticker)
    add_to_queue(ticker)


# using yfinance (try, except is for error management)
def get_current_price(ticker):
    data_close = yf.Ticker(ticker).history(period="5d", interval="1m")
    try:
        current_price = data_close['Close'].iloc[-1]
        return current_price
    except:
        return -1


def trailing_stop_loss(ticker):  # returns True if you need to sell
    stock_price_index = 0
    for stock in our_stocks_list:
        if stock == ticker:
            break
        stock_price_index += 1

    try:  # checks to see if we are missing any prices in comparison to the stock list
        price = our_prices_list[stock_price_index]
        pre_own = True
    except:
        pre_own = False

    current_stock_price = get_current_price(ticker)
    if current_stock_price == -1:  # error management (sells the stock)
        our_stocks_list.remove(ticker)
        our_prices_list.remove(our_prices_list[stock_price_index])
        return True

    if pre_own is True:
        max_stock_price = our_prices_list[stock_price_index]
        if current_stock_price > max_stock_price:
            our_prices_list[stock_price_index] = current_stock_price
            return False
        elif current_stock_price < (max_stock_price*(1-trailing_stop_loss_constant)):
            our_stocks_list.remove(ticker)
            our_prices_list.remove(our_prices_list[stock_price_index])
            return True
    else:
        our_prices_list.append(current_stock_price)
        return False


def add_to_queue(ticker):  # sell code
    queue_list = []
    new_in_queue = ticker + " SELL"

    with open('recordsqueue.txt', 'r') as filehandle:  # grabs current watchlist
        for line in filehandle:
            in_queue = line[:-1]
            queue_list.append(in_queue)

    queue_list.append(new_in_queue)

    with open('recordsqueue.txt', 'w') as file:  # updates with new stock
        for obj in queue_list:
            file.write(obj + "\n")


schedule.every().day.at("10:10").do(main_SELL_function)
schedule.every().day.at("16:00").do(schedule.clear)

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
            schedule.every(1).minutes.do(main_SELL_function)
            first_iteration = False
    # Keep the script running
    schedule.run_pending()
    t.sleep(1)