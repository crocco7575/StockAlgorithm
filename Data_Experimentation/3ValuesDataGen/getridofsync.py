import pandas as pd

file_path= 'combined_file.xlsx'
df = pd.read_excel(file_path)

# Drop the last column
df = df.iloc[:, :-1]

# Save the updated file
df.to_excel(file_path, index=False)
print("done")