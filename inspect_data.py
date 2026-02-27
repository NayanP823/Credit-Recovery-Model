import pandas as pd

try:
    df = pd.read_excel("default of credit card clients.xls", header=1)
    print("Columns:", df.columns.tolist())
    print("First few rows:\n", df.head())
except Exception as e:
    print(f"Error reading file: {e}")
