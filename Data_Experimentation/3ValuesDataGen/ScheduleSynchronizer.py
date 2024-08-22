import pandas as pd

ticker_list = []

with open('condensedtickers.txt', 'r') as filehandle:
    for line in filehandle:
        ind_ticker = line[:-1]
        ticker_list.append(ind_ticker)


def synchronize(my_ticker, index, dataf):

    my_flag_time = dataf.loc[index, 'Flag Time']

    ticker_index = 0
    for ticker in ticker_list:
        if ticker == my_ticker:
            break
        ticker_index += 1

    base_time_list = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    real_times = []

    if ticker_index <= 1760:
        for t in base_time_list:
            real_times.append(t)
    elif 1761 <= ticker_index <= 3520:
        for t in base_time_list:
            real_times.append(t + 5)
    elif ticker_index > 3520:
        for t in base_time_list:
            real_times.append(t + 10)

    match = False
    for t in real_times:
        if t == my_flag_time:
            match = True
            break

    return match


def update_excel(file):
    df = pd.read_excel(file)
    df['Sync'] = [None] * len(df)

    ind = 0
    for ticker in df['Ticker']:

        my_pl = df.loc[ind, 'P/L ($)']

        result = synchronize(ticker, ind, df)
        if result:
            df.loc[ind, 'Sync'] = my_pl
        else:
            df.loc[ind, 'Sync'] = 0

        ind += 1
    df.to_excel(file, index=False)


sl_list = ['0.0205', '0.0255', '0.0305', '0.0355', '0.0405']
date_list = ['2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17']

for spacer in sl_list:
    for day in date_list:
        file_path = 'records_folder/trailing_sl/' + spacer + '/' + day + '/record.xlsx'
        update_excel(file_path)

    print(spacer + ' complete')
