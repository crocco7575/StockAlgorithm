import pandas as pd

trial_list = ['buy_spacer', 'normal_sl']
buy_spacer_list = ['1.0003', '1.0008', '1.0013', '1.0018', '1.0023', '1.0028', '1.0033', '1.0038', '1.0043', '1.0048',
                   '1.0053', '1.0058', '1.0063', '1.0068', '1.0073', '1.0078', '1.0083', '1.0088', '1.0093', '1.0098',
                   '1.0103', '1.0108', '1.0113', '1.0118', '1.0123', '1.0153', '1.0163', '1.0168', '1.0173',]
sl_list = ['0.0025', '0.0075', '0.0125', '0.0175', '0.0225']
date_list = ['2024-05-17', '2024-05-16', '2024-05-15', '2024-05-14', '2024-05-13']
# Read the first Excel file

for param in trial_list:
    if param == 'buy_spacer':
        for spacer in buy_spacer_list:
            date_index = 0
            for date in date_list:
                if date == '2024-05-13':
                    continue

                file_path_1 = 'records_folder/buy_spacer/' + spacer + '/' + date + '/record.xlsx'
                file_path_2 = 'records_folder/buy_spacer/' + spacer + '/' + date_list[date_index + 1] + '/record.xlsx'
                df1 = pd.read_excel(file_path_1)

                # Read the second Excel file
                df2 = pd.read_excel(file_path_2)

                # Identify the common rows
                common_rows = df1.merge(df2, how='inner')

                # Remove the common rows from df1
                df1_cleaned = df1[~df1.apply(tuple, 1).isin(common_rows.apply(tuple, 1))]

                # Save the cleaned DataFrame to a new Excel file
                df1_cleaned.to_excel(file_path_1, index=False)
                date_index += 1
            print(spacer + ' complete')

    elif param == 'normal_sl':
        for sl in sl_list:
            date_index = 0
            for date in date_list:
                if date == '2024-05-13':
                    continue

                file_path_1 = 'records_folder/normal_sl/' + sl + '/' + date + '/record.xlsx'
                file_path_2 = 'records_folder/normal_sl/' + sl + '/' + date_list[date_index + 1] + '/record.xlsx'
                df1 = pd.read_excel(file_path_1)

                # Read the second Excel file
                df2 = pd.read_excel(file_path_2)

                # Identify the common rows
                common_rows = df1.merge(df2, how='inner')

                # Remove the common rows from df1
                df1_cleaned = df1[~df1.apply(tuple, 1).isin(common_rows.apply(tuple, 1))]

                # Save the cleaned DataFrame to a new Excel file
                df1_cleaned.to_excel(file_path_1, index=False)
                date_index += 1
            print(sl + ' complete')
