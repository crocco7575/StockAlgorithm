from polygon import RESTClient
import pandas as pd

ticker_list = []

client = RESTClient(api_key='d5SH0oK1pY1XuuSY_Fpdo4RCuZH2RU4S')

with open('condensedtickers.txt', 'r') as filehandle:
    for line in filehandle:
        ind_ticker = line[:-1]
        ticker_list.append(ind_ticker)

resume = False

for ticker in ticker_list:

    if ticker == 'SQQQ':
        resume = True

    if not resume:
        print(ticker + ' complete')
        continue

    open_list = []
    high_list = []
    low_list = []
    close_list = []
    volume_list = []
    timestamp = []
    for ohlc in client.list_aggs(ticker, 1, 'minute', '2019-06-10', '2024-06-08',
                                 adjusted=True):
        open_list.append(ohlc.open)
        high_list.append(ohlc.high)
        low_list.append(ohlc.low)
        close_list.append(ohlc.close)
        volume_list.append(ohlc.volume)
        timestamp.append(ohlc.timestamp)

    data = {
        'Open': open_list,
        'High': high_list,
        'Low': low_list,
        'Close': close_list,
        'Volume': volume_list,
        'Timestamp': timestamp
    }
    df = pd.DataFrame(data)

    file_path = 'records_folder/daily_data/' + ticker + '/historical_data.xlsx'

    if len(df) > 1048575:
        print(ticker + ' too large')
        with pd.ExcelWriter(file_path) as writer:
            df1 = df.iloc[:1048575]
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2 = df.iloc[1048575:]
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
    else:
        df.to_excel(file_path, index=False)

    print(ticker + ' complete')




