import pandas as pd

def load_data(Crop_recommendation):
    df = pd.read_csv(Crop_recommendation)
    return df

def preprocess(df):
    # Example preprocessing (customize as needed)
    df = df.dropna()
    return df
