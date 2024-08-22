import pandas as pd

df1 = pd.read_excel('records_folder/ml_data/data_file2.xlsx')
df2 = pd.read_excel('records_folder/ml_data/data_file3.xlsx')
df3 = pd.read_excel('records_folder/ml_data/data_file4.xlsx')

combined_df = pd.concat([df1, df2, df3], ignore_index=True)
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

shuffled_df.to_excel('records_folder/ml_data/ml_train_file1.xlsx', index=False)
