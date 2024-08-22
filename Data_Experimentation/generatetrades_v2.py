from talipp.ohlcv import OHLCVFactory
import talipp.indicators as ta
import datetime
import pandas as pd

buy_spacer = 1.0126
normal_stop_loss_constant = 0.0443
trailing_stop_loss_constant = 0.028

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


def main_flag_func():
    for ticker in ticker_list:

        file_path = 'records_folder/daily_data/' + ticker + '/historical_data.xlsx'
        df1 = pd.read_excel(file_path, sheet_name='Sheet1')
        try:
            df2 = pd.read_excel(file_path, sheet_name='Sheet2')
            df3 = pd.concat([df1, df2])
        except ValueError:
            df3 = df1

        price_one = clean(df3)
        price_five = price_by_5min(price_one)

        op = price_five['Open'].to_list()
        high = price_five['High'].to_list()
        low = price_five['Low'].to_list()
        close = price_five['Close'].to_list()
        vol = price_five['Volume'].to_list()

        price_five['RSI'] = ta.RSI(14, close)
        price_five = create_ttm(price_five, op, high, low, close, vol)

        dfs = []
        prev_val = None
        temp_df = pd.DataFrame()
        for index, row in price_one.iterrows():
            if prev_val is not None and row['Minutes'] < prev_val:
                dfs.append(temp_df)
                temp_df = pd.DataFrame()
            temp_df = pd.concat([temp_df, row.to_frame().T])
            prev_val = row['Minutes']
        dfs.append(temp_df)

        dfs_5min = []
        prev_val = None
        temp_df = pd.DataFrame()
        for index, row in price_five.iterrows():
            if prev_val is not None and row['Minutes'] < prev_val:
                dfs_5min.append(temp_df)
                temp_df = pd.DataFrame()
            temp_df = pd.concat([temp_df, row.to_frame().T])
            prev_val = row['Minutes']
        dfs_5min.append(temp_df)

        date_list = []
        for frame in dfs_5min:
            format_date =

        print('Num of days: ' + str(len(dfs)))
        print('Num of days (5 min): ' + str(len(dfs_5min)))

        for i, day_dataframe in enumerate(dfs):

            price_five_inner = dfs_5min[i]

            c = price_five_inner['Close'].to_list()
            t = price_five_inner['Minutes'].to_list()
            rsi = price_five_inner['RSI'].to_list()
            ttm = price_five_inner['TTM'].to_list()

            price_index = 0
            current_index = 0

            for current_price in c:
                if current_index < price_index:
                    current_index += 1
                    continue
                elif current_index == price_index:
                    current_index += 1

                if ttm[price_index] is None:
                    price_index += 1
                    continue

                rsi_result = check_rsi(price_index, rsi)
                ttm_result = check_ttm(price_index, ttm)
                if rsi_result == "invalid" or rsi_result is None or ttm_result == "invalid" or ttm_result is None:
                    price_index += 1
                    continue
                if rsi_result[0] == "buy" and ttm_result[0] == "buy":
                    print("flag")
                    minute_index = main_watch_func(ticker, float(current_price), t[price_index], rsi_result[1],
                                                   ttm_result[1], price_one)

                    try:
                        if (minute_index[0] == 'removed from watchlist' or minute_index[0] ==
                                'traded' or minute_index == 'day end'):
                            for time in t[price_index:]:
                                if time >= minute_index[1]:
                                    price_index += 1
                                    break
                                price_index += 1
                            continue
                        else:
                            print(ticker + " broke")
                    except TypeError:
                        print(ticker + " broke")
                        break

                price_index += 1

        convert_to_excel(ticker)
        global df
        df = pd.DataFrame(columns=df.columns)
        print(ticker + ' complete')


def main_watch_func(ticker, flag_price, flag_timer, buy_rsi, buy_ttm, price_by_1min):

    c = price_by_1min['Close'].to_list()
    t = price_by_1min['Minutes'].to_list()

    base_index = 0
    for time_tracker in t:
        if time_tracker > flag_timer:
            break
        base_index += 1

    price_index = base_index

    if price_index == len(t) - 1:
        return ['day end', t[price_index]]

    for current_price in c[price_index:]:

        if t[price_index] - flag_timer > 20:
            return ['removed from watchlist', t[price_index]]

        if float(current_price) >= float(buy_spacer * flag_price):
            if t[price_index] < flag_timer:
                price_index += 1
                continue
            new_row = {'Ticker': ticker, 'Flag Price': flag_price, 'Buy Price': float(current_price),
                       'Flag Time': flag_timer, 'Buy Time': t[price_index], 'Sell Time': 0,
                       'Time Between Flag and Buy': (t[price_index] - flag_timer),
                       'Lowest RSI': buy_rsi, 'TTM Strength': buy_ttm, 'P/L ($)': 0, 'P/L (%)': 0
                       }
            df.loc[len(df)] = new_row
            return main_sell_func(float(current_price), c, price_index, t)

        price_index += 1


def main_sell_func(buy_price, price_by_1min, list_index, time_list):
    price_index = list_index
    max_price = buy_price
    for current_price in price_by_1min[(price_index + 1):]:

        if price_index == len(price_by_1min[(price_index + 1):]) - 1:
            return sell_action(buy_price, float(current_price), price_index, time_list)

        if float(current_price) > max_price:
            max_price = float(current_price)
        if (float(current_price) - buy_price) < 0:
            if float(current_price) < (buy_price * (1 - normal_stop_loss_constant)):
                return sell_action(buy_price, float(current_price), price_index, time_list)
        else:
            if current_price < (max_price * (1 - trailing_stop_loss_constant)):
                return sell_action(buy_price, float(current_price), price_index, time_list)

        price_index += 1


def sell_action(buy_price, sell_price, list_index, time_list):
    pl_pct = ((sell_price - buy_price) / buy_price) * 100
    quantity = int(3000 / buy_price)
    pl_dlr = ((quantity * buy_price) * (1 + (pl_pct / 100))) - (quantity * buy_price)

    df.at[len(df) - 1, 'P/L ($)'] = pl_dlr
    df.at[len(df) - 1, 'P/L (%)'] = pl_pct

    df.at[len(df) - 1, 'Sell Time'] = time_list[list_index]
    return ['traded', time_list[list_index]]


def create_ttm(dataframe, o, h, l, c, v):

    ohlcv = OHLCVFactory.from_matrix2([
        o,
        h,
        l,
        c,
        v
    ])

    dataframe['TTMVal'] = ta.TTM(20, input_values=ohlcv)
    ttm = []
    for val in dataframe['TTMVal']:
        if val is None:
            ttm.append(None)
        else:
            ttm.append(val.histogram)

    dataframe['TTM'] = ttm
    dataframe.drop(columns='TTMVal')

    return dataframe


def check_rsi(list_index, rsi):
    if list_index < 2:
        return "invalid"

    try:
        validity = True
        if rsi[list_index] is None or rsi[list_index - 1] is None or rsi[list_index - 2] is None:
            validity = False
        if validity:
            past_rsi = [float(rsi[list_index]), float(rsi[list_index - 1]), float(rsi[list_index - 2])]
        else:
            return "invalid"
    except IndexError or TypeError:
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


def check_ttm(list_index, ttm):
    if list_index < 3:
        return 'invalid'

    validity = True
    for item in ttm:
        if item is None:
            validity = False
    if validity:
        try:
            slope_a = ttm[list_index - 2] - ttm[list_index - 3]
            slope_b = ttm[list_index - 1] - ttm[list_index - 2]
            slope_c = ttm[list_index] - ttm[list_index - 1]
            if slope_a < 0 and slope_b > 0 and slope_c > 0:
                return ["buy", float(ttm[list_index] - ttm[list_index - 2])]
            else:
                return None
        except IndexError:
            return "invalid"
    else:
        return "invalid"


def clean(dataframe):

    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'], unit='ms')
    dataframe = dataframe[(dataframe['Timestamp'].dt.time >= datetime.time(9, 30)) &
                          (dataframe['Timestamp'].dt.time <= datetime.time(16, 0))]

    minutes_list = []
    for t in dataframe['Timestamp']:
        t1 = str(t).split(' ')
        t2 = t1[1].split(':')
        minutes_list.append((int(t2[0]) * 60 + int(t2[1])) - 570)

    dataframe['Minutes'] = minutes_list

    return dataframe


def price_by_5min(dataframe):
    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
    dataframe['minute'] = dataframe['Timestamp'].dt.minute
    dataframe['even_5_min'] = dataframe['minute'] % 5 == 0

    new_df = dataframe[dataframe['even_5_min']]
    new_df = new_df.drop(columns=['minute', 'even_5_min'])

    return new_df


def download_tickers():
    with open('condensedtickers.txt', 'r') as filehandle:
        for line in filehandle:
            ind_ticker = line[:-1]
            ticker_list.append(ind_ticker)


def convert_to_excel(ticker):
    global df
    df.to_excel('records_folder/daily_data/' + ticker + '/trade_record_v2.xlsx', index=False)


download_tickers()
main_flag_func()

