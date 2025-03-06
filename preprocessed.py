import pandas as pd

for company in ['AMZN', 'AAPL', 'MSFT', 'TSLA']:
    df = pd.read_csv(f"{company}.csv")

    df["senti_label"] = df["senti_label"] == "bullish"
    df.to_csv(f"{company}.csv")