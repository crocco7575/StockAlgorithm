import yfinance as yf
from talipp.ohlcv import OHLCVFactory
import talipp.indicators as ta
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
import matplotlib.pyplot as plt

buy_spacer = 1.0158
normal_stop_loss_constant = 0.0025
trailing_stop_loss_constant = 0.003

data = {
    'Ticker': [],
    'Flag Price': [],
    'Buy Price': [],
    'Flag Time': [],
    'Buy Time': [],
    'Sell Time': [],
    'Time Between Flag and Buy': [],
    'Lowest RSI': [],
    'TTM Strength': [],
    'P/L ($)': [],
    'P/L (%)': []
}

df = pd.DataFrame(data)

ticker_list = []


def main_flag_func(five_days, day_after, test_date):
    for ticker in ticker_list:
        data_set = yf.Ticker(ticker)

        try:
            data_history = data_set.history(start=test_date, end=next_day_str, interval='5m')
            price_by_min = data_history['Close'].tolist()
        except IndexError:
            continue

        price_index = 0
        current_index = 0

        for current_price in price_by_min:
            if current_index < price_index:
                current_index += 1
                continue
            elif current_index == price_index:
                current_index += 1

            rsi_result = check_rsi(ticker, price_index, five_days, day_after)
            ttm_result = check_ttm(ticker, price_index, five_days, day_after)
            if rsi_result == "invalid" or rsi_result is None or ttm_result == "invalid" or ttm_result is None:
                price_index += 1
                continue
            if rsi_result[0] == "buy" and ttm_result[0] == "buy":
                minute_index = main_watch_func(ticker, float(current_price), time_in_min(price_index), price_index,
                                               rsi_result[1], ttm_result[1], test_date)

                try:
                    if minute_index == 'removed from watchlist':
                        price_index += 4
                        continue
                    else:
                        price_index = math.ceil(int(minute_index) / 5)
                        continue
                except TypeError:
                    print(ticker + " broke")
                    break

            price_index += 1


def main_watch_func(ticker, flag_price, flag_timer, list_index, buy_rsi, buy_ttm, test_date):
    data_set = yf.Ticker(ticker)
    data_history = data_set.history(start=test_date, end=next_day_str, interval='1m')
    price_by_1min = data_history['Close'].tolist()

    price_index = (list_index * 5) + 4
    time_reference = []
    if len(price_by_1min) != 389:
        new_time_by_min = data_history.index.tolist()
        for timestamp in new_time_by_min:
            new = str(timestamp).split(',')
            new2 = new[0].split(' ')
            new3 = new2[1].split('-')
            minutes_seconds = new3[0].split(':')
            time_final = (int(minutes_seconds[0]) * 60) + int(minutes_seconds[1]) - 570
            time_reference.append(time_final)

        start_index = 0
        for time_min in time_reference:
            if price_index <= time_min:
                break
            start_index += 1

        price_index = start_index

    if len(price_by_1min[(price_index + 1):]) == 0:
        return len(price_by_1min) - 1

    for current_price in price_by_1min[(price_index + 1):]:

        if len(price_by_1min) == 389:
            if time_in_1min(price_index) - flag_timer > 20:
                return 'removed from watchlist'
        else:
            if time_reference[price_index] - flag_timer > 20:
                return 'removed from watchlist'
            elif price_index == (len(price_by_1min) - 2):
                return 'removed from watchlist'

        if float(current_price) >= float(buy_spacer * flag_price):
            if len(price_by_1min) == 389:
                new_row = {'Ticker': ticker, 'Flag Price': flag_price, 'Buy Price': float(current_price),
                           'Flag Time': flag_timer, 'Buy Time': time_in_1min(price_index), 'Sell Time': 0,
                           'Time Between Flag and Buy': (time_in_1min(price_index) - flag_timer), 'Lowest RSI': buy_rsi,
                           'TTM Strength': buy_ttm, 'P/L ($)': 0, 'P/L (%)': 0
                           }
                df.loc[len(df)] = new_row
                return main_sell_func(float(current_price), price_by_1min, price_index, time_reference)
            else:
                new_row = {'Ticker': ticker, 'Flag Price': flag_price, 'Buy Price': float(current_price),
                           'Flag Time': flag_timer, 'Buy Time': time_reference[price_index], 'Sell Time': 0,
                           'Time Between Flag and Buy': (time_reference[price_index] - flag_timer),
                           'Lowest RSI': buy_rsi, 'TTM Strength': buy_ttm, 'P/L ($)': 0, 'P/L (%)': 0
                           }
                df.loc[len(df)] = new_row
                return main_sell_func(float(current_price), price_by_1min, price_index, time_reference)

        price_index += 1


def main_sell_func(buy_price, price_by_1min, list_index, time_list):
    price_index = list_index
    max_price = buy_price
    for current_price in price_by_1min[(list_index + 1):]:
        if float(current_price) > max_price:
            max_price = float(current_price)
        if (float(current_price) - buy_price) < 0:
            if float(current_price) < (buy_price * (1 - normal_stop_loss_constant)):
                return sell_action(buy_price, float(current_price), price_index, time_list)
        else:
            if current_price < (max_price * (1 - trailing_stop_loss_constant)):
                return sell_action(buy_price, float(current_price), price_index, time_list)

        if price_index == len(price_by_1min) - 2:
            return sell_action(buy_price, float(current_price), price_index, time_list)

        price_index += 1


def sell_action(buy_price, sell_price, list_index, time_list):
    pl_pct = ((sell_price - buy_price) / buy_price) * 100
    quantity = int(3000 / buy_price)
    pl_dlr = ((quantity * buy_price) * (1 + (pl_pct / 100))) - (quantity * buy_price)

    df.at[len(df) - 1, 'P/L ($)'] = pl_dlr
    df.at[len(df) - 1, 'P/L (%)'] = pl_pct

    if len(time_list) == 0:
        df.at[len(df) - 1, 'Sell Time'] = time_in_1min(list_index)
        return list_index
    else:
        df.at[len(df) - 1, 'Sell Time'] = time_list[list_index]
        return time_list[list_index]


def download_tickers():
    with open('condensedtickers.txt', 'r') as filehandle:
        for line in filehandle:
            ind_ticker = line[:-1]
            ticker_list.append(ind_ticker)


def rsi_values(ticker, five_days, day_after):
    power = yf.Ticker(ticker)
    try:
        df = power.history(start=five_days, end=day_after, interval="5m")
        df['RSI'] = ta.RSI(14, df['Close'])
        rsi_list = df['RSI'].tail(80).tolist()
        return rsi_list
    except IndexError:
        return "invalid"


# function to calculate TTM squeeze histogram value
def ttm_squeeze_values(ticker, five_days, day_after):
    data_set = yf.Ticker(ticker)
    data_history = data_set.history(start=five_days, end=day_after, interval='5m')
    ohlcv = OHLCVFactory.from_matrix2([
        data_history['Open'],
        data_history['High'],
        data_history['Low'],
        data_history['Close'],
        data_history['Volume']
    ])

    data_history['TTM'] = ta.TTM(20, input_values=ohlcv)
    ttm_values_list = []
    for ttm in data_history['TTM'].tail(81):
        if ttm is None:
            return "invalid"
        ttm_values_list.append(ttm.histogram)
    return ttm_values_list


def check_rsi(ticker, list_index, five_days, day_after):
    rsi_index = list_index + 2
    day_rsi = rsi_values(ticker, five_days, day_after)

    if day_rsi == "invalid":
        return "invalid"

    try:
        past_rsi = [float(day_rsi[rsi_index]), float(day_rsi[rsi_index - 1]), float(day_rsi[rsi_index - 2])]
    except IndexError:
        return "invalid"

    rsi_under_30 = False
    lowest_rsi = past_rsi[0]
    for rsi in past_rsi:
        if rsi < 30:
            rsi_under_30 = True
        if rsi < lowest_rsi:
            lowest_rsi = rsi

    if rsi_under_30:
        return ["buy", lowest_rsi]
    else:
        return None


# checks TTM squeeze value to return "buy" "sell" or "invalid"
def check_ttm(ticker, list_index, five_days, day_after):
    ttm_index = list_index + 3

    day_ttm = ttm_squeeze_values(ticker, five_days, day_after)
    if day_ttm == "invalid":
        return "invalid"
    validity = True
    for item in day_ttm:
        if item is None:
            validity = False
    if validity:
        try:
            slope_a = day_ttm[ttm_index - 2] - day_ttm[ttm_index - 3]
            slope_b = day_ttm[ttm_index - 1] - day_ttm[ttm_index - 2]
            slope_c = day_ttm[ttm_index] - day_ttm[ttm_index - 1]
            if slope_a < 0 and slope_b > 0 and slope_c > 0:
                return ["buy", float(day_ttm[ttm_index] - day_ttm[ttm_index - 2])]
            else:
                return None
        except IndexError:
            return "invalid"
    else:
        return "invalid"


def time_in_min(list_index):
    return 5 * (list_index + 1)


def time_in_1min(list_index):
    return list_index + 1


def convert_to_excel(test_date):
    global df

    excel_file_path = ('records_folder/' + 'buy_spacer/' + str(buy_spacer) + '/' + test_date + '/' + 'record.xlsx')

    # Write DataFrame to Excel
    df.to_excel(excel_file_path, index=False)  # Set index=False to exclude the DataFrame index from the Excel file

    print("DataFrame has been written to Excel successfully.")


def create_graphs(test_date):
    global df

    x_list = ['TTM Strength', 'Lowest RSI', 'Flag Time', 'Time Between Flag and Buy', 'Flag Price']
    ref_list = ['ttm', 'rsi', 'flag_time', 'time_betw', 'flag_price']

    ref_index = 0
    for x_value in x_list:
        fig, ax = plt.subplots()
        # Plot the data as a scatter plot
        ax.scatter(df[f'{x_value}'], df['P/L (%)'], c=['green' if p > 0 else 'red' for p in df['P/L (%)']])

        # Set labels and title
        ax.set_xlabel(f'{x_value}')
        ax.set_ylabel('Profit/Loss Percent')
        ax.set_title(f'Profit/Loss Percent vs {x_value}')
        # Set the x-axis to start at 0
        ax.set_xlim(left=0)
        # Set the x-axis to zero
        ax.spines['bottom'].set_position('zero')
        plt.title(f'P/L vs {x_value}')
        plt.savefig('records_folder/' + 'buy_spacer/' + str(buy_spacer) + '/' + test_date + '/' + ref_list[ref_index] +
                    '.png')
        plt.clf()
        ref_index += 1


download_tickers()

test_date_list = ['2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17']

for day in test_date_list:
    test_date_dt = datetime.strptime(day, '%Y-%m-%d')
    test_date_np = np.datetime64(test_date_dt, 'D')
    five_business_days_before_np = np.busday_offset(test_date_np, -4, roll='backward')
    five_business_days_before_dt = five_business_days_before_np.astype(datetime)
    next_day_dt = test_date_dt + timedelta(days=1)

    five_business_days_before_str = five_business_days_before_dt.strftime('%Y-%m-%d')
    next_day_str = next_day_dt.strftime('%Y-%m-%d')

    main_flag_func(five_business_days_before_str, next_day_str, day)
    convert_to_excel(day)
    create_graphs(day)