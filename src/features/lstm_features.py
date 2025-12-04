from src.config import DATE_COLUMN, USER_ID_COLUMN, TRUE_VARS
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def create_cyclical_feature(df: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    df = df.copy()

    df["day_of_week"] = df[DATE_COLUMN].dt.weekday  # 0=pon, 6=niedz
    df["day_of_month"] = df[DATE_COLUMN].dt.day     # 1-31

    #cyclical model of the weeek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    #cyclical model of the month
    df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31.0)
    df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31.0)
    
    return df

def create_sequences(df:pd.DataFrame, time_steps:int = 10,feature_columns:list[str] = TRUE_VARS, target_column: str = "migraine_tomorrow") -> tuple:
    sequences = []
    targets = []
    
    for user_id, group in df.groupby(USER_ID_COLUMN):
        group = group.sort_values(DATE_COLUMN)  # Sort by date
        for i in range(len(group) - time_steps):
            X = group[feature_columns].iloc[i:i + time_steps].values  # Sequence of features
            y = group[target_column].iloc[i + time_steps]  # Target (migraine_tomorrow) for the next day
            sequences.append(X)
            targets.append(y)
    
    X_data = np.array(sequences)
    y_data = np.array(targets)

    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)

    return X_data, y_data, scaler