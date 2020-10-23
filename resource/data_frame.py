import pandas as pd

def read_df():
    df = pd.read_csv("https://pycourse.s3.amazonaws.com/temperature.csv")
    df.sort_values
    return df
