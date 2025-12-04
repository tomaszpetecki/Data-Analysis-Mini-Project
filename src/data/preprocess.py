from __future__ import annotations
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import src.features.lstm_features as ftrs

#RAW DATA ---> PREPROCESSED DATA
#we need to think about what we need, although the data does not contain any missing values,
#just to demonstrate a good practice of a "data scientist", and to be toally sure that we are not 
#learning on dirty data, I will clean them

from src.config import USER_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN, RANDOM_SEED, VARS
from src.data.load_data import load_raw_migraine_data

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    #We need to remove potential duplicates of (user,data)
    df = df.copy()
    df = df.drop_duplicates([USER_ID_COLUMN,DATE_COLUMN])

    #Remove all the rows with NaN
    df = df.dropna(axis=0)
    
    #We set the index to datetime object
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values([USER_ID_COLUMN,DATE_COLUMN]).reset_index(drop=True)
    return df

#Function for finding the episodes in data where target satisfies a condition (by default target == 1).
#The function returns a list of tuples with (start,end) for each episode in terms of column X in the df

#Function for truncating all groups to be of the same length
def truncate(df:pd.DataFrame, threshold : int|None= None, group_col: str = USER_ID_COLUMN) ->pd.DataFrame:
    "truncates each group to satisfy the threshold, otherwise it will use the minimum number of rows per group as a threshold"
    "truncate and preprocess"
    #For LSTM we will need to learn from equal-length sequences, so we can truncate to the minimum entry
    #The threshold argument is only for cases when we want to
    df = df.copy()
    #Check the minimum value of entries per user
    n = df.groupby(group_col).size().min() if threshold is None else threshold
    #Now truncate each group entry to that size
    df_first_n = df.groupby('user_id', group_keys=False).apply(lambda x: x.head(n)).reset_index(drop=True)
    return df_first_n

#Adds 
def add_tomorrow_label(df: pd.DataFrame,source_col: str | None = None,label_name: str = "migraine_tomorrow") -> pd.DataFrame:
    """
    we basically shift the column migraine occurence to make it our target - migraine tommorow. There will be NaN in the last row.
    We need to prepare some features for the binary classification - 1 migraine occurs tommorow, 0 - no migraine tomorrow
    """
    df = df.copy()

    col = source_col if source_col is not None else TARGET_COLUMN

    #Shift tye migraine occurence by 1, so that it represents target "migraine tommorow" 
    df[label_name] = (df.groupby(USER_ID_COLUMN)[col].shift(-1))

    #In the last row there always will be a NaN as there is no tommorow values when shifting
    df = df.dropna(subset=[label_name]).reset_index(drop=True)

    # We get the type 0,1
    df[label_name] = df[label_name].astype(int)

    return df

def preprocess(df) -> pd.DataFrame:
    "Cleans data and adds tomorrow label, without truncation"
    df = df.copy()
    df = clean_data(df)
    df = add_tomorrow_label(df)

    return df

def fully_preprocess(df, options="lstm-truncate") -> pd.DataFrame:
    "additionally adds features"
    df = df.copy()
    df = preprocess(df)
    s = options.lower().strip().split("-")
    if "lstm" in s:
        df = ftrs.create_cyclical_feature(df)
        if "truncate" in s:
            df = truncate(df)
    return df


if __name__ == "__main__":
    from src.features.lstm_features import create_sequences
    from src.config import PROCESSED_DATA_DIR
    df = preprocess(load_raw_migraine_data())
    df = truncate(df)
    X, y,_ = create_sequences(df,feature_columns= VARS + ["dow_sin","dow_cos","dom_sin","dom_cos"])
    print(X.shape)
    #data_frame = pd.DataFrame({"X":X,"y":y})
    #data_frame.to_csv(PROCESSED_DATA_DIR / "training_data.csv",index=False)

    #print(df["sleep_hours"].quantile([0.25,0.75]))
