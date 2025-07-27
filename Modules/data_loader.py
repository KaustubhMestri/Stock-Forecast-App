import pandas as pd

def load_data(file_path="Data/Historical_data.csv"):
    """Load and preprocess the historical stock data."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.transpose()
    df.index = [f"Day -{100-i}" for i in range(len(df))]
    print(df.head())
    return df

