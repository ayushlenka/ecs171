import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv("CSV_Files/AMZN.csv")

# print(df.iloc[:,  [4] + list(range(6, 18))])

for name in ['AMZN', 'AAPL', 'MSFT', 'TSLA']:
    df = pd.read_csv(f'CSV_Files/{name}.csv')

    print(df.iloc[:, 4])