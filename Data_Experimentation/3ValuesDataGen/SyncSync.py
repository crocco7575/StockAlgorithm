import pandas as pd

buy_spacer_list = ['0.0205', '0.0255', '0.0305', '0.0355', '0.0405']
date_list = ['2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17']

data = {
        'Buy Spacer': [0.0205, 0.0255, 0.0305, 0.0355, 0.0405]
        }
df_main = pd.DataFrame(data)
df_main['P/L Avg'] = [None] * len(df_main)

spacer_ind = 0
for spacer in buy_spacer_list:
    sum = 0

    for day in date_list:
        file_path = 'records_folder/trailing_sl/' + spacer + '/' + day + '/record.xlsx'
        df = pd.read_excel(file_path)

        for pl in df['Sync']:
            sum += pl
    avg_sum = sum/5

    df_main.loc[spacer_ind, 'P/L Avg'] = avg_sum

    print(spacer + ' complete')
    spacer_ind += 1

file = 'records_folder/trailing_sl/' + 'average3.xlsx'
df_main.to_excel(file, index=False)


