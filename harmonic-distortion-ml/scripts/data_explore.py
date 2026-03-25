import pandas as pd
import numpy as np

file_path = 'all_signals_1000_1.csv'
df = pd.read_csv(file_path, header=None)

with open('data_info.txt', 'w') as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Data types: {df.dtypes.unique()}\n")
    f.write(f"Missing values: {df.isnull().sum().sum()}\n")
    f.write(f"First 5 values of first row: {df.iloc[0, :5].tolist()}\n")
    f.write(f"First 5 values of last row: {df.iloc[-1, :5].tolist()}\n")
    f.write(f"Mean of first row: {df.iloc[0].mean()}\n")
    f.write(f"Std of first row: {df.iloc[0].std()}\n")
    
print("Data analysis written to data_info.txt")
