import pandas as pd

def Read_CSV(filename):
    return pd.read_csv(filename)

def Write_CSV(data, filename):
    data.to_csv(filename, index=False)