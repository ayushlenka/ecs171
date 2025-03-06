import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("CSV_Files/merged_file.csv")

for name in ['AMZN', 'AAPL', 'MSFT', 'TSLA']:
    df = data[data["ticker"] == name]
    df.drop(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume", "processed"], inplace=True)

    df1 = pd.get_dummies(df["emo_label"], prefix="emo")
    df = pd.concat([df, df1], axis=1)
    df.drop(columns=["emo_label"], inplace=True)
    df.reset_index(drop=True,inplace=True)

    # print(df)

    df.to_csv(f"CSV_Files/{name}.csv")