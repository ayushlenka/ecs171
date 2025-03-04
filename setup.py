import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

file_paths = {
    "AAPL": "./aapl_df.csv",
    "AMZN": "./AMZN.csv",
    "GOOGL": "./GOOG_SENTIMENT.csv",
    "MSFT": "./processed_msft_data.csv",
    "TSLA": "./TSLA.csv",
}

dataframes = {company: pd.read_csv(path) for company, path in file_paths.items()}

def preprocess_data(df):
    emotion_encoder = OneHotEncoder(sparse_output=False)
    emotion_encoded = emotion_encoder.fit_transform(df[['emo_label']])
    emotion_df = pd.DataFrame(emotion_encoded, columns=emotion_encoder.categories_[0])
    
    sentiment_encoder = OneHotEncoder(sparse_output=False)
    sentiment_encoded = sentiment_encoder.fit_transform(df[['senti_label']])
    sentiment_df = pd.DataFrame(sentiment_encoded, columns=sentiment_encoder.categories_[0])

    df = pd.concat([df, emotion_df, sentiment_df], axis=1)
    df.drop(columns=['emo_label', 'senti_label'], inplace=True)

    return df

dataframes = {company: preprocess_data(df) for company, df in dataframes.items()}
