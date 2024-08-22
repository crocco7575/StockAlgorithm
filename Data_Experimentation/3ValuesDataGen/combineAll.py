import pandas as pd

# Load the two Excel files into DataFrames
combined_df = pd.read_excel('records_folder/buy_spacer/1.0003/2024-05-13/record.xlsx', sheet_name='Sheet1')
buy_spacer_list = ['1.0008', '1.0013', '1.0018', '1.0023', '1.0028', '1.0033', '1.0038', '1.0043', '1.0048',
                   '1.0053', '1.0058', '1.0063', '1.0068', '1.0073', '1.0078', '1.0083', '1.0088', '1.0093', '1.0098']
date_list = ['2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17']

for spacer in buy_spacer_list:
    for day in date_list:
        if spacer == '1.0098' and day == '2024-05-17':
            print(spacer + ' ' + day + ' skipped')
            continue
        file = 'records_folder/buy_spacer/' + spacer + '/' + day + '/record.xlsx'
        df2 = pd.read_excel(file, sheet_name='Sheet1')
        combined_df = pd.concat([combined_df, df2], ignore_index=True)
    print(spacer + ' complete')

combined_df.to_excel('records_folder/buy_spacer/combined_file.xlsx', index=False)
