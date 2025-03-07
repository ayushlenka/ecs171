import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("CSV_Files/MSFT.csv")

df.drop(columns=["id"], inplace=True)
print(df)

df.to_csv("CSV_Files/MSFT.csv", index=False)