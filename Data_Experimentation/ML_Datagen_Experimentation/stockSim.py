import pandas as pd
import numpy as np
import random
from talipp.ohlcv import OHLCVFactory
import talipp.indicators as ta
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

buy_spacer = 1.0126
normal_stop_loss_constant = 0.0443
trailing_stop_loss_constant = 0.028


def bell_curve(x, translation):
    sigma = 1.5
    y = np.exp(-(((x-translation)-5)**2) / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return y


def scale_mapping(n):
    if 1 <= n <= 10:
        return -5 + (n * 8 / 10)
    else:
        return "Input must be between 1 and 10 (inclusive)"


def scale_mapping_2(n):
    if 1 <= n <= 10:
        return 110 - (n * 10)
    else:
        return "Input must be between 1 and 10 (inclusive)"


def scale_mapping_3(n):
    return np.where((n >= 0) & (n <= 0.05), np.minimum(1 + (n / 0.005), 10), np.nan)


def check_rsi(index, rsi_list):
    past_3_rsi = [rsi_list[index - 2], rsi_list[index - 1], rsi_list[index]]
    rsi_under_30 = False
    lowest_rsi = 30
    for value in past_3_rsi:
        if value < 30:
            rsi_under_30 = True
        if value < lowest_rsi:
            lowest_rsi = value

    if rsi_under_30:
        return ["buy", float(lowest_rsi)]
    else:
        return None


def check_ttm(index, ttm_list):
    slope_a = ttm_list[index - 2] - ttm_list[index - 3]
    slope_b = ttm_list[index - 1] - ttm_list[index - 2]
    slope_c = ttm_list[index] - ttm_list[index - 1]
    if slope_a < 0 and slope_b > 0 and slope_c > 0:
        return ["buy", float(ttm_list[index] - ttm_list[index - 2])]
    else:
        return None


# order: open, close, volume, high, low

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

update_excel = 100
day_number = 0

with open('pctdictionary.txt', 'r') as filehandle:
    pctdistro_dictionary = [line.strip() for line in filehandle]

first_iter = True
target_trades = 0

for ratings in pctdistro_dictionary:
    new_ratings = ratings.split(",")

    volatility_rating = int(new_ratings[0])

    price_rating = int(new_ratings[1])

    number_of_trades = round(float(new_ratings[2]) * 500000)

    print(f"Currently executing Vol: {volatility_rating}, Price: {price_rating}, for {number_of_trades} trades")

    if first_iter:
        target_trades = number_of_trades
        first_iter = False
    else:
        target_trades += number_of_trades

    while len(df) < (target_trades - 34202):
        starting_price = round(random.uniform(((price_rating - 1) * 8.6), (price_rating * 8.6)), 2)

        o = np.zeros(1950)
        h = np.zeros(1950)
        l = np.zeros(1950)
        c = np.zeros(1950)
        v = np.zeros(1950)

        for ind in range(1950):
            # OPEN value assigned
            o[ind] = starting_price if ind == 0 else c[ind - 1]

            # CLOSE value calculated and assigned
            trans = scale_mapping(volatility_rating)
            divisor = scale_mapping_2(volatility_rating)
            final_pick_list = (np.arange(0.25, 50.25, 0.25) / divisor).repeat(
                (bell_curve((np.arange(0.25, 50.25, 0.25) / 5), trans) * 375.99639).astype(int))
            final_pct_index = np.random.randint(0, len(final_pick_list) - 1)
            final_pct = final_pick_list[final_pct_index]

            # CLOSE positive or negative change
            pos_or_neg = np.random.randint(1, 3, size=(1,)) * 2 - 3
            close_value = o[ind] * (1 + (final_pct / 100) * pos_or_neg)
            c[ind] = close_value

            # HIGH/LOW
            high_low_pct_list = (np.arange(0.0002, 0.05, 0.0002)).repeat(
                (bell_curve(scale_mapping_3(np.arange(0.0002, 0.05, 0.0002)), trans) * 375.99639).astype(int))
            final_high_index = np.random.randint(0, len(high_low_pct_list) - 1)
            final_low_index = np.random.randint(0, len(high_low_pct_list) - 1)
            final_high = high_low_pct_list[final_high_index]
            final_low = high_low_pct_list[final_low_index]

            if pos_or_neg > 0:
                h[ind] = c[ind] * (1 + final_high)
                l[ind] = o[ind] * (1 - final_low)
            else:
                h[ind] = o[ind] * (1 + final_high)
                l[ind] = c[ind] * (1 - final_low)

            # VOLUME calculation
            hl_diff = abs((h[ind] - l[ind]) / h[ind])
            v[ind] = hl_diff * 500000

        # Vectorized creation of 5-minute OHLCV data
        o5 = o[::5]
        h5 = h[::5]
        l5 = l[::5]
        c5 = c[::5]
        v5 = v[::5]

        # Vectorized creation of RSI and TTM lists
        ohlcv = OHLCVFactory.from_matrix2([o5, h5, l5, c5, v5])
        rsi = ta.RSI(14, c5)
        ttm_og = ta.TTM(20, input_values=ohlcv)
        ttm = np.array([val.histogram if val is not None else None for val in ttm_og])

        # Append to lists
        one_minute_price = c[-390:]
        five_minute_price = c5[-78:]
        five_minute_rsi = rsi[-78:]
        five_minute_ttm = ttm[-78:]

        # fig, axs = plt.subplots(3)

        # axs[0].plot(c5[-78:])
        # axs[0].set_ylabel('Price')

        # axs[1].plot(rsi[-78:])
        # axs[1].axhline(y=70, color='r', linestyle='-')
        # axs[1].axhline(y=30, color='g', linestyle='-')
        # axs[1].set_ylabel('RSI')

        # axs[2].bar(range(len(ttm[-78:])), ttm[-78:])
        # axs[2].set_ylabel('TTM')

        # plt.show()

        skip_iterations = 0
        for price_index, curr_price in enumerate(five_minute_price):
            if skip_iterations > 0:
                skip_iterations -= 1
                continue

            if price_index < 3 or price_index >= 360:
                continue

            rsi_result = check_rsi(price_index, five_minute_rsi)
            ttm_result = check_ttm(price_index, five_minute_ttm)
            if rsi_result is None or ttm_result is None:
                continue

            if rsi_result[0] == "buy" and ttm_result[0] == "buy":
                one_minute_price_index = (price_index * 5) + 1
                flag_timer_five = price_index
                flag_price = curr_price

                for current_price in one_minute_price[one_minute_price_index:]:
                    if one_minute_price_index - (flag_timer_five * 5) == 20:
                        skip_iterations = 4
                        break

                    if one_minute_price_index >= 360:
                        skip_iterations = int((one_minute_price_index / 5) - price_index)
                        break

                    if float(current_price) >= float(buy_spacer * flag_price):
                        new_row = {'Ticker': "MLSIM", 'Flag Price': flag_price, 'Buy Price': float(current_price),
                                'Flag Time': flag_timer_five * 5, 'Buy Time': one_minute_price_index, 'Sell Time': 0,
                                'Time Between Flag and Buy': (one_minute_price_index - flag_timer_five * 5),
                                'Lowest RSI': rsi_result[1], 'TTM Strength': ttm_result[1], 'P/L ($)': 0, 'P/L (%)': 0
                                }
                        df.loc[len(df)] = new_row

                        buy_price = float(current_price)
                        max_price = buy_price
                        for sell_price in one_minute_price[one_minute_price_index:]:
                            if one_minute_price_index == 385:  # sells at 15:55 every day
                                pl_pct = ((sell_price - buy_price) / buy_price) * 100
                                quantity = int(3000 / buy_price)
                                pl_dlr = ((quantity * buy_price) * (1 + (pl_pct / 100))) - (quantity * buy_price)
                                df.at[len(df) - 1, 'P/L ($)'] = pl_dlr
                                df.at[len(df) - 1, 'P/L (%)'] = pl_pct
                                df.at[len(df) - 1, 'Sell Time'] = one_minute_price_index
                                skip_iterations = int((one_minute_price_index / 5) - price_index)
                                break

                            if float(sell_price) > max_price:  # update max price
                                max_price = float(sell_price)
                            if (float(sell_price) - buy_price) < 0:
                                if float(sell_price) < (buy_price * (1 - normal_stop_loss_constant)):
                                    pl_pct = ((sell_price - buy_price) / buy_price) * 100
                                    quantity = int(3000 / buy_price)
                                    pl_dlr = ((quantity * buy_price) * (1 + (pl_pct / 100))) - (quantity * buy_price)
                                    df.at[len(df) - 1, 'P/L ($)'] = pl_dlr
                                    df.at[len(df) - 1, 'P/L (%)'] = pl_pct
                                    df.at[len(df) - 1, 'Sell Time'] = one_minute_price_index
                                    skip_iterations = int((one_minute_price_index / 5) - price_index)
                                    break
                            else:
                                if sell_price < (max_price * (1 - trailing_stop_loss_constant)):
                                    pl_pct = ((sell_price - buy_price) / buy_price) * 100
                                    quantity = int(3000 / buy_price)
                                    pl_dlr = ((quantity * buy_price) * (1 + (pl_pct / 100))) - (quantity * buy_price)
                                    df.at[len(df) - 1, 'P/L ($)'] = pl_dlr
                                    df.at[len(df) - 1, 'P/L (%)'] = pl_pct
                                    df.at[len(df) - 1, 'Sell Time'] = one_minute_price_index
                                    skip_iterations = int((one_minute_price_index / 5) - price_index)
                                    break
                            one_minute_price_index += 1
                        break
                    one_minute_price_index += 1

        if len(df) > update_excel:
            df.to_excel('records_folder/ml_data/data_file4.xlsx', index=False)
            update_excel += 1000
            print(len(df))

    print(f"Finished Vol: {volatility_rating}, Price: {price_rating}, with {len(df)} trades")

df.to_excel('records_folder/ml_data/data_file4.xlsx', index=False)

# average trades per day = 0.104
# code started at 10:26 PM

# A: 0.000000
# B: 0.001018
# C: 0.001000
# D: 0.000000
# E: 0.000000
# F: 0.007531



